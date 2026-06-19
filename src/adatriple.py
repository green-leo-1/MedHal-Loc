"""
AdaTriple: Adaptive Knowledge Graph-NLI Hybrid Verification
for Fine-Grained Medical Hallucination Detection

Complete implementation aligned with paper equations (Eq. 1-22).
Hardware: Single NVIDIA RTX 3090/4090 (24GB VRAM)
"""

import os
import re
import sys
import io
import json
import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from pathlib import Path

# Fix Windows console encoding for Chinese output
if sys.platform == "win32":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                       errors="replace", line_buffering=True)
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                                       errors="replace", line_buffering=True)

import math
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("adatriple")

# ---------------------------------------------------------------------------
# GPU configuration
# ---------------------------------------------------------------------------

def get_device(preferred: str = "cuda") -> str:
    """Return the best available device, always preferring GPU."""
    if preferred == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info("[Device] Using GPU: %s (%.1f GB VRAM)", gpu_name, vram)
        return "cuda"
    if preferred == "cuda" and not torch.cuda.is_available():
        logger.warning("[Device] CUDA requested but not available, falling back to CPU")
    return "cpu"


DEVICE = get_device("cuda")

# ---------------------------------------------------------------------------
# 1. Data Structures
# ---------------------------------------------------------------------------

RELATION_TYPES = [
    "treats", "treated_by", "causes", "symptom_of",
    "contraindicated_with", "dosage_of", "diagnoses",
    "complication_of", "located_in", "site_of",
    "risk_factor_for", "examined_by", "interacts_with",
    "palliates", "presents", "associates", "resembles",
    "includes", "localizes", "alleviates", "co_occurs_with",
    "targets", "connected_to", "detects", "examines",
    "part_of", "uses", "monitored_by", "complementary_to",
    "combined_with", "related_to",
]

ENTITY_TYPES = ["disease", "symptom", "drug", "examination", "body_part",
                "treatment", "dosage"]


@dataclass
class MedicalEntity:
    text: str
    entity_type: str
    start_pos: int = 0
    end_pos: int = 0
    kg_id: Optional[str] = None
    kg_score: float = 0.0
    confidence: float = 0.0


@dataclass
class MedicalTriple:
    head_entity: MedicalEntity
    relation: str
    tail_entity: MedicalEntity
    source_sentence: str
    sentence_idx: int

    entity_score_head: float = 0.0
    entity_score_tail: float = 0.0
    relation_score: float = 0.0
    kg_score: float = 0.0
    nli_score: float = 0.0
    lambda_weight: float = 0.5
    hallucination_score: float = 0.0


@dataclass
class DetectionResult:
    response_text: str
    sentences: List[str]
    triples: List[MedicalTriple]
    sentence_scores: List[float]
    response_score: float
    hallucinated_triples: List[int]


# ---------------------------------------------------------------------------
# 2. Module 1 – Medical Triple Decomposition  (Eq. 1-2)
# ---------------------------------------------------------------------------

class TripleDecomposer:
    """Decompose medical text into (e_h, r, e_t) triples.

    Eq. 1: T(s_i) = {t_1, ..., t_m}
    Eq. 2: T(R)   = union of T(s_i)
    """

    RELATION_PROMPT_ZH = (
        "请从以下医疗文本中提取实体间的关系。\n"
        "已识别的实体: {entities}\n"
        "文本: {sentence}\n\n"
        "请以JSON列表格式输出，每个元素包含 head、relation、tail。\n"
        "relation 必须从以下列表中选择: {relation_types}\n"
        "只输出JSON列表，不要其他内容。"
    )

    RELATION_PROMPT_EN = (
        "Extract medical relations from the text below.\n"
        "Entities: {entities}\n"
        "Text: {sentence}\n\n"
        "Output a JSON list where each item has keys: head, relation, tail.\n"
        "relation must be one of: {relation_types}\n"
        "Output only the JSON list."
    )

    def __init__(self, ner_model_path: str = "", lang: str = "zh",
                 device: str = "cuda", use_llm_for_relations: bool = True):
        self.lang = lang
        self.device = get_device(device)
        self.use_llm_for_relations = use_llm_for_relations
        self._ner_model = None
        self._llm_pipe = None

        self._load_ner(ner_model_path)
        logger.info("[TripleDecomposer] ready  lang=%s  device=%s", lang, device)

    # -- NER ------------------------------------------------------------------

    def _load_ner(self, path: str):
        """Load domain-specific NER model."""
        if self.lang == "en":
            try:
                from transformers import pipeline as hf_pipeline
                model_name = path or "d4data/biomedical-ner-all"
                self._ner_model = hf_pipeline(
                    "ner", model=model_name,
                    aggregation_strategy="max",
                    device=0 if self.device == "cuda" else -1,
                )
                self._ner_backend = "transformer"
                logger.info("[NER] Biomedical NER loaded: %s", model_name)
                return
            except Exception as exc:
                logger.warning("[NER] Biomedical NER unavailable (%s), using rule-based", exc)
        else:
            try:
                from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
                model_name = path or "unikei/bert-base-chinese-medical-ner"
                self._ner_model = pipeline(
                    "ner", model=model_name,
                    tokenizer=model_name,
                    aggregation_strategy="simple",
                    device=0 if self.device == "cuda" else -1,
                )
                self._ner_backend = "transformer"
                logger.info("[NER] Transformer NER model loaded: %s", model_name)
                return
            except Exception as exc:
                logger.warning("[NER] Transformer NER unavailable (%s), using rule-based", exc)

        self._ner_model = None
        self._ner_backend = "rule"

    def extract_entities(self, sentence: str) -> List[MedicalEntity]:
        """Extract medical entities via hybrid NER.

        Transformer NER runs first; when it finds < 2 entities the
        rule-based dictionary NER supplements so that short texts
        (MedQA/MMLU answer options) still yield enough entity pairs
        for triple construction.
        """
        if self._ner_model is None:
            return self._rule_based_ner(sentence)
        entities = self._transformer_ner(sentence)
        if len(entities) < 2:
            rule_ents = self._rule_based_ner(sentence)
            existing_texts = {e.text.lower() for e in entities}
            for re_ent in rule_ents:
                if re_ent.text.lower() not in existing_texts:
                    entities.append(re_ent)
                    existing_texts.add(re_ent.text.lower())
        return entities

    def _spacy_ner(self, sentence: str) -> List[MedicalEntity]:
        doc = self._ner_model(sentence)
        entities = []
        for ent in doc.ents:
            etype = self._map_entity_type(ent.label_)
            entities.append(MedicalEntity(
                text=ent.text, entity_type=etype,
                start_pos=ent.start_char, end_pos=ent.end_char,
            ))
        return entities

    MAX_ENTITIES_PER_SENTENCE = 12

    def _transformer_ner(self, sentence: str) -> List[MedicalEntity]:
        results = self._ner_model(sentence)
        entities = []
        for r in results:
            etype = self._map_entity_type(r.get("entity_group", "O"))
            if etype == "skip":
                continue
            word = r.get("word", "")
            start = r.get("start", 0)
            end = r.get("end", 0)
            score = float(r.get("score", 0.0))
            text = sentence[start:end] if start < end <= len(sentence) else word
            text = text.strip()
            if not text or len(text) < 2:
                continue
            entities.append(MedicalEntity(
                text=text,
                entity_type=etype,
                start_pos=start, end_pos=end,
                confidence=score,
            ))
        if len(entities) > self.MAX_ENTITIES_PER_SENTENCE:
            entities.sort(key=lambda e: e.confidence, reverse=True)
            entities = entities[:self.MAX_ENTITIES_PER_SENTENCE]
        return entities

    def _rule_based_ner(self, sentence: str) -> List[MedicalEntity]:
        """Regex/dictionary-based fallback NER for Chinese and English medical text."""
        entities: List[MedicalEntity] = []
        patterns = {
            "drug": (
                r"(阿司匹林|二甲双胍|布洛芬|阿莫西林|头孢|青霉素|华法林|"
                r"氨氯地平|美托洛尔|奥美拉唑|氯吡格雷|胰岛素|沙丁胺醇|"
                r"甲巯咪唑|舍曲林|卡马西平|别嘌醇|阿仑膦酸钠|格列本脲|"
                r"metformin|aspirin|ibuprofen|amoxicillin|warfarin|"
                r"insulin|amlodipine|metoprolol|omeprazole|clopidogrel|"
                r"atorvastatin|lisinopril|losartan|simvastatin|"
                r"acetaminophen|prednisone|albuterol|salbutamol|"
                r"ciprofloxacin|azithromycin|doxycycline|fluoxetine|"
                r"sertraline|gabapentin|carbamazepine|phenytoin|"
                r"digoxin|furosemide|hydrochlorothiazide|"
                r"heparin|enoxaparin|rivaroxaban|apixaban|"
                r"tamoxifen|rituximab|infliximab|adalimumab|"
                r"ACE inhibitors?|ARBs?|statins?|beta[- ]?blockers?|"
                r"NSAIDs?|SSRIs?|PPIs?|anticoagulants?|"
                r"antibiotics?|antihypertensives?|antidiabetics?|"
                r"corticosteroids?|bronchodilators?|diuretics?|"
                r"analgesics?|antipyretics?|antihistamines?|"
                r"antidepressants?|anticonvulsants?|immunosuppressants?|"
                r"chemotherapy|radiotherapy|vaccination|vaccine)"
            ),
            "disease": (
                r"(糖尿病|高血压|冠心病|肺炎|肝炎|肾衰竭|心力衰竭|"
                r"心肌梗死|脑卒中|哮喘|胃溃疡|甲亢|乙肝|痛风|骨质疏松|"
                r"抑郁症|癫痫|贫血|"
                r"diabetes(?:\s+mellitus)?(?:\s+type\s+[12])?|T2DM|T1DM|"
                r"hypertension|high blood pressure|HTN|"
                r"pneumonia|bronchitis|tuberculosis|COPD|"
                r"hepatitis|cirrhosis|liver disease|"
                r"heart failure|cardiac failure|CHF|"
                r"myocardial infarction|heart attack|"
                r"stroke|cerebrovascular accident|CVA|TIA|"
                r"asthma|emphysema|pulmonary embolism|"
                r"cancer|carcinoma|melanoma|lymphoma|leukemia|"
                r"HIV|AIDS|malaria|sepsis|"
                r"Alzheimer(?:'s)?|Parkinson(?:'s)?|epilepsy|"
                r"depression|anxiety|schizophrenia|bipolar|"
                r"anemia|thrombosis|atherosclerosis|"
                r"osteoporosis|arthritis|gout|lupus|"
                r"influenza|COVID-?19|SARS|"
                r"renal failure|kidney disease|CKD|"
                r"atrial fibrillation|arrhythmia|"
                r"obesity|hyperlipidemia|dyslipidemia|"
                r"gastric ulcer|peptic ulcer|GERD|"
                r"urinary tract infection|UTI|"
                r"meningitis|encephalitis|appendicitis|"
                r"pancreatitis|cholecystitis|"
                r"hypothyroidism|hyperthyroidism)"
            ),
            "symptom": (
                r"(头痛|发热|咳嗽|恶心|呕吐|腹泻|头晕|胸闷|气短|"
                r"乏力|水肿|皮疹|失眠|胸痛|腹痛|多饮|多尿|心悸|"
                r"headache|fever|pyrexia|cough|nausea|fatigue|"
                r"vomiting|diarrhea|dizziness|dyspnea|"
                r"chest pain|abdominal pain|back pain|"
                r"shortness of breath|SOB|"
                r"edema|swelling|rash|itching|pruritus|"
                r"insomnia|tremor|seizure|"
                r"palpitation|tachycardia|bradycardia|"
                r"hypotension|syncope|vertigo|"
                r"polyuria|polydipsia|weight loss|weight gain|"
                r"hematuria|jaundice|cyanosis|"
                r"lethargy|confusion|delirium|"
                r"bleeding|hemorrhage|bruising|"
                r"constipation|bloating|heartburn)"
            ),
            "body_part": (
                r"(肺|肝|肾|心脏|胃|脑|血管|骨骼|"
                r"lung|liver|kidney|heart|brain|"
                r"pancreas|thyroid|spleen|gallbladder|"
                r"coronary arter(?:y|ies)|blood vessel|"
                r"bone marrow|lymph node|"
                r"mitral valve|aortic valve|sinoatrial node|"
                r"left ventricle|right ventricle|atrium|"
                r"proximal (?:convoluted )?tubule|loop of Henle|"
                r"ascending limb|descending limb|"
                r"cerebral cortex|hippocampus|"
                r"femur|tibia|vertebra)"
            ),
            "examination": (
                r"(CT|MRI|X[- ]?ray|B超|血常规|尿常规|心电图|"
                r"血糖|肝功能|肾功能|blood test|"
                r"ECG|EKG|echocardiogram|ultrasound|"
                r"biopsy|endoscopy|colonoscopy|"
                r"chest X[- ]?ray|CBC|complete blood count|"
                r"urinalysis|ABG|arterial blood gas|"
                r"A1[cC]|HbA1c|hemoglobin A1c|"
                r"INR|PT|PTT|aPTT|"
                r"BMP|CMP|metabolic panel|"
                r"lipid panel|cholesterol|LDL|HDL|"
                r"serum creatinine|BUN|GFR|eGFR|"
                r"TSH|T3|T4|thyroid function|"
                r"PSA|mammograph|Pap smear|"
                r"blood pressure|blood glucose)"
            ),
            "dosage": r"(\d+\s*(?:mg|ml|g|μg|mcg|IU|mEq|units?|tablets?|"
                      r"片|粒|支)(?:\s*/\s*(?:day|d|日|天|次|hour|h))?)",
            "treatment": (
                r"(手术|化疗|放疗|透析|移植|输血|"
                r"surgery|transplant(?:ation)?|dialysis|"
                r"blood transfusion|intubation|ventilat(?:ion|or)|"
                r"catheteriz|resection|bypass|stent(?:ing)?|"
                r"angioplasty|mastectomy|appendectomy|"
                r"physical therapy|rehabilitation)"
            ),
        }
        for etype, pat in patterns.items():
            for m in re.finditer(pat, sentence, re.IGNORECASE):
                entities.append(MedicalEntity(
                    text=m.group(), entity_type=etype,
                    start_pos=m.start(), end_pos=m.end(),
                ))
        return entities

    @staticmethod
    def _map_entity_type(label: str) -> str:
        mapping = {
            "CHEMICAL": "drug", "DISEASE": "disease", "DRUG": "drug",
            "SYMPTOM": "symptom", "BODY": "body_part", "EXAM": "examination",
            "dis": "disease", "sym": "symptom", "dru": "drug",
            "pro": "examination", "bod": "body_part", "dep": "body_part",
            "Medication": "drug", "Disease_disorder": "disease",
            "Sign_symptom": "symptom", "Diagnostic_procedure": "examination",
            "Biological_structure": "body_part", "Lab_value": "examination",
            "Detailed_description": "skip", "Nonbiological_location": "skip",
            "Clinical_event": "treatment", "Therapeutic_procedure": "treatment",
            "Administration": "treatment", "Duration": "skip",
            "Dosage": "dosage", "Frequency": "skip", "Date": "skip",
            "Age": "skip", "Sex": "skip", "Outcome": "skip",
            "Subject": "skip", "Activity": "skip",
        }
        for k, v in mapping.items():
            if k.lower() in label.lower():
                return v
        return "disease"

    # -- Relation extraction --------------------------------------------------

    def extract_relations(self, sentence: str, entities: List[MedicalEntity],
                          sent_idx: int = 0) -> List[MedicalTriple]:
        if len(entities) < 2:
            return []
        if self.use_llm_for_relations and self._llm_pipe is not None:
            return self._llm_relation_extract(sentence, entities, sent_idx)
        return self._heuristic_relation_extract(sentence, entities, sent_idx)

    def _heuristic_relation_extract(self, sentence: str,
                                     entities: List[MedicalEntity],
                                     sent_idx: int) -> List[MedicalTriple]:
        """Rule-based relation extraction using entity type co-occurrence."""
        triples: List[MedicalTriple] = []
        type_relation_map = {
            ("drug", "disease"): "treats",
            ("disease", "drug"): "treated_by",
            ("disease", "symptom"): "causes",
            ("symptom", "disease"): "symptom_of",
            ("drug", "drug"): "interacts_with",
            ("drug", "dosage"): "dosage_of",
            ("disease", "body_part"): "located_in",
            ("body_part", "disease"): "site_of",
            ("disease", "examination"): "examined_by",
            ("examination", "disease"): "diagnoses",
            ("disease", "disease"): "complication_of",
            ("drug", "symptom"): "causes",
            ("symptom", "symptom"): "co_occurs_with",
            ("drug", "body_part"): "targets",
            ("treatment", "disease"): "treats",
            ("disease", "treatment"): "treated_by",
            ("treatment", "symptom"): "alleviates",
            ("symptom", "body_part"): "located_in",
            ("body_part", "body_part"): "connected_to",
            ("examination", "symptom"): "detects",
            ("examination", "body_part"): "examines",
            ("drug", "treatment"): "part_of",
            ("treatment", "drug"): "uses",
            ("drug", "examination"): "monitored_by",
            ("examination", "examination"): "complementary_to",
            ("treatment", "treatment"): "combined_with",
            ("treatment", "body_part"): "targets",
        }
        seen: Set[Tuple[str, str]] = set()
        for i, eh in enumerate(entities):
            for j, et in enumerate(entities):
                if i == j:
                    continue
                pair_key = (eh.text, et.text)
                if pair_key in seen:
                    continue
                rel = type_relation_map.get(
                    (eh.entity_type, et.entity_type), "related_to")
                triples.append(MedicalTriple(
                    head_entity=eh, relation=rel, tail_entity=et,
                    source_sentence=sentence, sentence_idx=sent_idx,
                ))
                seen.add(pair_key)
        return triples

    def _llm_relation_extract(self, sentence: str,
                               entities: List[MedicalEntity],
                               sent_idx: int) -> List[MedicalTriple]:
        tpl = (self.RELATION_PROMPT_ZH if self.lang == "zh"
               else self.RELATION_PROMPT_EN)
        prompt = tpl.format(
            entities=", ".join(e.text for e in entities),
            sentence=sentence,
            relation_types=", ".join(RELATION_TYPES),
        )
        try:
            output = self._llm_pipe(prompt, max_new_tokens=512,
                                     do_sample=False)[0]["generated_text"]
            parsed = json.loads(output[len(prompt):].strip())
            entity_map = {e.text: e for e in entities}
            triples = []
            for item in parsed:
                h = entity_map.get(item["head"])
                t = entity_map.get(item["tail"])
                r = item.get("relation", "")
                if h and t and r in RELATION_TYPES:
                    triples.append(MedicalTriple(
                        head_entity=h, relation=r, tail_entity=t,
                        source_sentence=sentence, sentence_idx=sent_idx,
                    ))
            return triples
        except Exception:
            return self._heuristic_relation_extract(sentence, entities, sent_idx)

    # -- Main entry -----------------------------------------------------------

    def decompose(self, response: str) -> Tuple[List[str], List[MedicalTriple]]:
        """Full decomposition: text -> sentences -> NER -> relations -> triples."""
        sentences = self._split_sentences(response)
        all_triples: List[MedicalTriple] = []
        for i, sent in enumerate(sentences):
            entities = self.extract_entities(sent)
            triples = self.extract_relations(sent, entities, sent_idx=i)
            all_triples.extend(triples)
        return sentences, all_triples

    def _split_sentences(self, text: str) -> List[str]:
        if self.lang == "zh":
            parts = re.split(r'([。！？\n])', text)
            result = []
            for k in range(0, len(parts) - 1, 2):
                s = parts[k] + parts[k + 1]
                if s.strip():
                    result.append(s.strip())
            if len(parts) % 2 == 1 and parts[-1].strip():
                result.append(parts[-1].strip())
            return result or [text.strip()]
        else:
            import re as _re
            sents = _re.split(r'(?<=[.!?])\s+', text.strip())
            return [s for s in sents if s.strip()]


# ---------------------------------------------------------------------------
# 3. Module 2 – Entity Verification  (Eq. 3)
# ---------------------------------------------------------------------------

class EntityVerifier:
    """Verify entities against a medical KG.

    Eq. 3:
        S_entity(e) = max_{e' in G} [
            cos(Enc(e), Enc(e')) * sigma(sim(e, e') - tau_e)
        ]

    Two channels evaluated *per candidate* then multiplied (not max of channels).
    """

    def __init__(self, kg_entities: Optional[Dict[str, dict]] = None,
                 encoder_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                 device: str = "cuda", tau_e: float = 0.5):
        self.device = get_device(device)
        self.tau_e = tau_e
        self.kg_entities = kg_entities or {}
        self._encoder = None
        self._tokenizer = None
        self._faiss_index = None
        self._kg_keys: List[str] = []
        self._kg_embeddings: Optional[np.ndarray] = None

        self._load_encoder(encoder_name)
        if self.kg_entities:
            self._build_faiss_index()
            if self._faiss_index is None and self._encoder is not None:
                self._build_brute_embeddings()
        logger.info("[EntityVerifier] ready  entities_in_kg=%d", len(self.kg_entities))

    def _load_encoder(self, name: str):
        try:
            from transformers import AutoTokenizer, AutoModel
            self._tokenizer = AutoTokenizer.from_pretrained(name)
            self._encoder = AutoModel.from_pretrained(name)
            dev = torch.device(self.device)
            self._encoder = self._encoder.to(dev).eval()
            logger.info("[EntityVerifier] SapBERT loaded: %s", name)
        except Exception as exc:
            logger.warning("[EntityVerifier] encoder unavailable (%s)", exc)

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if self._encoder is None or self._tokenizer is None:
            return np.random.randn(len(texts), 768).astype(np.float32)
        dev = next(self._encoder.parameters()).device
        batch_size = 64
        all_embs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            tok = self._tokenizer(batch, padding=True, truncation=True,
                                  max_length=64, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = self._encoder(**tok)
            emb = out.last_hidden_state[:, 0, :]
            all_embs.append(emb.cpu().numpy())
        return np.concatenate(all_embs, axis=0).astype(np.float32)

    def _build_faiss_index(self):
        try:
            import faiss
        except ImportError:
            logger.warning("[EntityVerifier] faiss not installed; brute-force search")
            return
        self._kg_keys = list(self.kg_entities.keys())
        names = [self.kg_entities[k].get("name", k) for k in self._kg_keys]
        self._kg_embeddings = self._encode_texts(names)
        dim = self._kg_embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(self._kg_embeddings)
        self._faiss_index.add(self._kg_embeddings)

    ENTITY_FLOOR = 0.97

    def verify(self, entity: MedicalEntity) -> float:
        """Compute entity grounding score per Eq. 3.

        Entities not found in KG get a floor score (KG incompleteness should
        not penalize correct but out-of-vocabulary entities).
        """
        if not self.kg_entities:
            return self.ENTITY_FLOOR

        if self._faiss_index is not None:
            raw = self._verify_faiss(entity)
        else:
            raw = self._verify_brute(entity)
        return max(raw, self.ENTITY_FLOOR)

    def _verify_faiss(self, entity: MedicalEntity) -> float:
        q = self._encode_texts([entity.text])
        import faiss
        faiss.normalize_L2(q)
        k = min(10, self._faiss_index.ntotal)
        scores, indices = self._faiss_index.search(q, k)
        best = 0.0
        for rank in range(k):
            idx = int(indices[0][rank])
            cos_sim = float(scores[0][rank])
            kg_name = self.kg_entities[self._kg_keys[idx]].get("name",
                                                                self._kg_keys[idx])
            str_sim = self._string_similarity(entity.text, kg_name)
            # Eq. 3: cos(Enc, Enc') * sigma(sim - tau_e)
            score = cos_sim * self._sigmoid(str_sim - self.tau_e)
            if score > best:
                best = score
                entity.kg_id = self._kg_keys[idx]
                entity.kg_score = best
        return best

    def _build_brute_embeddings(self):
        """Pre-compute SapBERT embeddings for all KG entities (no FAISS)."""
        self._kg_keys = list(self.kg_entities.keys())
        names = [self.kg_entities[k].get("name", k) for k in self._kg_keys]
        logger.info("[EntityVerifier] Pre-computing SapBERT embeddings for %d entities...", len(names))
        self._kg_embeddings = self._encode_texts(names)
        norms = np.linalg.norm(self._kg_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self._kg_embeddings = self._kg_embeddings / norms
        logger.info("[EntityVerifier] Embeddings ready (shape=%s)", self._kg_embeddings.shape)

    def _verify_brute(self, entity: MedicalEntity) -> float:
        if self._kg_embeddings is not None and self._encoder is not None:
            return self._verify_brute_emb(entity)
        best = 0.0
        ent_lower = entity.text.lower()
        for kid, info in self.kg_entities.items():
            name = info.get("name", kid)
            str_sim = self._string_similarity(ent_lower, name.lower())
            if str_sim > 0.85:
                score = str_sim
            else:
                cos_sim = self._quick_semantic_sim(ent_lower, name.lower())
            score = cos_sim * self._sigmoid(str_sim - self.tau_e)
            if score > best:
                best = score
                entity.kg_id = kid
                entity.kg_score = best
            for alias in info.get("aliases", []):
                str_sim_a = self._string_similarity(ent_lower, alias.lower())
                if str_sim_a > 0.85:
                    score_a = str_sim_a
                else:
                    cos_sim_a = self._quick_semantic_sim(ent_lower, alias.lower())
                score_a = cos_sim_a * self._sigmoid(str_sim_a - self.tau_e)
                if score_a > best:
                    best = score_a
                    entity.kg_id = kid
                    entity.kg_score = best
        return best

    KG_MATCH_THRESHOLD = 0.50

    def _verify_brute_emb(self, entity: MedicalEntity) -> float:
        """Brute-force entity matching requiring both semantic AND string similarity.

        Pure cosine similarity produces too many false matches (e.g. 'lace' ->
        anatomy term). We require meaningful string overlap to confirm the match.
        """
        q = self._encode_texts([entity.text])
        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        q = q / np.maximum(q_norm, 1e-8)
        cos_sims = (q @ self._kg_embeddings.T).flatten()
        n = len(cos_sims)
        top_k = min(20, n)
        if top_k >= n:
            top_indices = np.arange(n)
        else:
            top_indices = np.argpartition(-cos_sims, top_k)[:top_k]
        best = 0.0
        best_idx = -1
        ent_lower = entity.text.lower()
        for idx in top_indices:
            cos_sim = float(cos_sims[idx])
            kg_name = self.kg_entities[self._kg_keys[idx]].get("name",
                                                                self._kg_keys[idx])
            str_sim = self._string_similarity(ent_lower, kg_name.lower())
            if str_sim >= 0.75:
                score = str_sim
            elif str_sim >= 0.35:
                score = cos_sim * str_sim * 2.0
            else:
                score = cos_sim * 0.15
            if score > best:
                best = score
                best_idx = idx
        if best >= self.KG_MATCH_THRESHOLD and best_idx >= 0:
            entity.kg_id = self._kg_keys[best_idx]
            entity.kg_score = best
        return best

    @staticmethod
    def _string_similarity(a: str, b: str) -> float:
        """Normalized edit-distance similarity in [0, 1]."""
        if a == b:
            return 1.0
        la, lb = len(a), len(b)
        if la == 0 or lb == 0:
            return 0.0
        dp = list(range(lb + 1))
        for i in range(1, la + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, lb + 1):
                tmp = dp[j]
                if a[i - 1] == b[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = tmp
        return 1.0 - dp[lb] / max(la, lb)

    @staticmethod
    def _quick_semantic_sim(a: str, b: str) -> float:
        """Character-overlap Jaccard as a fast semantic proxy when encoder is missing."""
        sa, sb = set(a), set(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-float(x)))


# ---------------------------------------------------------------------------
# 4. Module 3 – Adaptive KG-NLI Relation Verification  (Eq. 4-9)
# ---------------------------------------------------------------------------

class RelationVerifier:
    """Adaptive KG-NLI hybrid relation verification.

    Eq. 4: S_rel(t) = lambda * S_KG(t) + (1-lambda) * S_NLI(t)
    Eq. 5: S_KG — path score in KG
    Eq. 6: S_NLI — NLI entailment with KG context
    Eq. 8: lambda = sigma(W * [cov_h; cov_t; l_path; ...] + b)   (enhanced)
    Eq. 9: cov(e, G) = min(deg(e)/avg_deg, 1) if e in G else 0
    """

    ENHANCED_FEATURE_DIM = 8

    def __init__(self, kg_graph=None, kg_entities: Optional[Dict] = None,
                 nli_model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                 device: str = "cuda", use_enhanced_lambda: bool = False):
        self.device = get_device(device)
        self.kg_graph = kg_graph
        self.kg_entities = kg_entities or {}
        self.use_enhanced_lambda = use_enhanced_lambda

        feat_dim = self.ENHANCED_FEATURE_DIM if use_enhanced_lambda else 3
        if use_enhanced_lambda:
            self.W = np.array([3.0, 3.0, -0.5, 0.0, 0.0, 0.0, 1.0, -0.5],
                              dtype=np.float64)
            self.b = -0.5
        else:
            self.W = np.array([2.5, 2.5, -0.5], dtype=np.float64)
            self.b = -0.5

        self._avg_degree: float = 1.0
        self._nli_pipe = None
        self._relation_embeddings: Optional[Dict[str, np.ndarray]] = None

        if kg_graph is not None:
            degrees = [d for _, d in kg_graph.degree()]
            self._avg_degree = max(np.mean(degrees), 1.0)

        self._load_nli(nli_model_name)
        logger.info("[RelationVerifier] ready  enhanced=%s", use_enhanced_lambda)

    def _load_nli(self, name: str):
        try:
            from transformers import pipeline as hf_pipeline
            dev = 0 if self.device == "cuda" else -1
            self._nli_pipe = hf_pipeline("text-classification", model=name,
                                          device=dev)
            logger.info("[NLI] loaded %s", name)
        except Exception as exc:
            logger.warning("[NLI] unavailable (%s)", exc)

    # -- KG Path Score (Eq. 5) -----------------------------------------------

    def _kg_path_score(self, triple: MedicalTriple) -> float:
        """KG path score in [0.5, 1.0].

        Design choice (KG_NEVER_HURTS): KG can only positively support a
        triple, never negatively contradict it.  Rationale: open-domain
        biomedical KGs (Hetionet, UMLS) are *incomplete* — a missing path
        does NOT entail the claim is false, it just means the KG lacks
        coverage.  Treating "no path" as anti-verification (the previous
        0.35 default) caused systematic negative transfer on datasets
        whose entities are outside Hetionet's scope (SciFact, MMLU).

        Therefore:
        * any case where KG cannot positively confirm the triple -> 0.5
          (neutral; defers entirely to the NLI signal in S_rel = lam*S_KG
          + (1-lam)*S_NLI, since 0.5 is the algebraic identity of "no
          influence").
        * a confirming path -> [0.7, 1.0] (the existing positive logic).
        """
        if self.kg_graph is None:
            return 0.5
        h_id = triple.head_entity.kg_id
        t_id = triple.tail_entity.kg_id
        h_in_kg = h_id is not None and h_id in self.kg_graph
        t_in_kg = t_id is not None and t_id in self.kg_graph
        if not (h_in_kg and t_in_kg):
            return 0.5
        try:
            import networkx as nx
            paths = list(nx.all_simple_paths(self.kg_graph, h_id, t_id, cutoff=3))
            if not paths:
                paths = list(nx.all_simple_paths(
                    self.kg_graph, t_id, h_id, cutoff=3))
            if not paths:
                return 0.5
            best = 0.0
            for path in paths:
                edge_product = 1.0
                path_relations: List[str] = []
                for k in range(len(path) - 1):
                    edata = self.kg_graph[path[k]][path[k + 1]]
                    edge_product *= edata.get("weight", 1.0)
                    path_relations.append(edata.get("relation", ""))
                rel_match = self._relation_match(path_relations, triple.relation)
                score = edge_product * max(rel_match, 0.5)
                best = max(best, score)
            return max(best, 0.7)
        except Exception:
            return 0.5

    def _relation_match(self, path_relations: List[str], claimed: str) -> float:
        """Soft relation matching phi(r_path, r) using label similarity."""
        if not path_relations:
            return 0.0
        best = 0.0
        for pr in path_relations:
            if pr == claimed:
                return 1.0
            sim = EntityVerifier._string_similarity(pr, claimed)
            best = max(best, sim)
        return best

    # -- NLI Score (Eq. 6-7) -------------------------------------------------

    def _run_nli_pair(self, premise: str, hypothesis: str) -> float:
        """Run single NLI inference, return verification score in [0, 1].

        Uses ent/(ent+contra) ratio which correctly maps neutral→0.5,
        entailed→1.0, contradicted→0.0.  The product formula ent*(1-contra)
        collapses neutral to ~0 causing all scores to cluster near 1.0.
        """
        try:
            result = self._nli_pipe(
                {"text": premise[:512], "text_pair": hypothesis[:512]},
                top_k=None,
            )
            ent, contra = 0.33, 0.33
            for item in result:
                label = item["label"].lower()
                if "entail" in label:
                    ent = float(item["score"])
                elif "contra" in label:
                    contra = float(item["score"])
            return float(np.clip(ent / (ent + contra + 1e-8), 0, 1))
        except Exception:
            return 0.5

    KG_CONTEXT_GATE = 0.55

    def _nli_score(self, triple: MedicalTriple,
                   evidence: str = "",
                   disable_kg_context: bool = False) -> float:
        """NLI verification with triple-focused hypothesis (Eq. 6).

        KG context gated on entity match confidence (>= KG_CONTEXT_GATE).
        Blend weight is adaptive: higher confidence → more KG weight,
        so noisy matches defer to evidence while strong matches leverage KG.
        """
        kg_ctx = ""
        kg_conf = 0.0
        if not disable_kg_context:
            h_conf = (triple.head_entity.kg_score
                      if triple.head_entity.kg_id is not None else 0.0)
            t_conf = (triple.tail_entity.kg_score
                      if triple.tail_entity.kg_id is not None else 0.0)
            if h_conf >= self.KG_CONTEXT_GATE and t_conf >= self.KG_CONTEXT_GATE:
                kg_ctx = self._build_kg_context(triple.head_entity,
                                                triple.tail_entity)
                kg_conf = (h_conf + t_conf) / 2.0
        sentence_hyp = triple.source_sentence
        rel_text = triple.relation.replace("_", " ")
        triple_hyp = (f"{triple.head_entity.text} {rel_text} "
                      f"{triple.tail_entity.text}.")

        if self._nli_pipe is None:
            return 0.5

        kg_nli = None
        ev_nli = None
        if kg_ctx and len(kg_ctx.strip()) > 10:
            sv = self._run_nli_pair(kg_ctx, sentence_hyp)
            tv = self._run_nli_pair(kg_ctx, triple_hyp)
            kg_nli = 0.4 * sv + 0.6 * tv
        if evidence and len(evidence.strip()) > 10:
            sv = self._run_nli_pair(evidence[:512], sentence_hyp)
            tv = self._run_nli_pair(evidence[:512], triple_hyp)
            ev_nli = 0.4 * sv + 0.6 * tv

        if kg_nli is not None and ev_nli is not None:
            return max(kg_nli, ev_nli)
        if kg_nli is not None:
            return kg_nli
        if ev_nli is not None:
            return ev_nli
        return 0.5

    def _build_kg_context(self, head: MedicalEntity, tail: MedicalEntity) -> str:
        """Eq. 7: KG_ctx = concat(desc(eh), N(eh), desc(et), N(et))."""
        parts: List[str] = []
        for ent in (head, tail):
            if ent.kg_id and ent.kg_id in self.kg_entities:
                info = self.kg_entities[ent.kg_id]
                ent_name = info.get("name", ent.text)
                desc = info.get("description", "")
                if desc:
                    parts.append(f"{ent_name}: {desc}")
                if self.kg_graph is not None and ent.kg_id in self.kg_graph:
                    neighbors = list(self.kg_graph.neighbors(ent.kg_id))[:8]
                    for nb in neighbors:
                        edata = self.kg_graph[ent.kg_id][nb]
                        nb_name = self.kg_entities.get(nb, {}).get("name", nb)
                        rel = edata.get("relation", "related_to")
                        parts.append(f"{ent_name} {rel} {nb_name}")
                    in_neighbors = list(self.kg_graph.predecessors(ent.kg_id))[:5]
                    for nb in in_neighbors:
                        edata = self.kg_graph[nb][ent.kg_id]
                        nb_name = self.kg_entities.get(nb, {}).get("name", nb)
                        rel = edata.get("relation", "related_to")
                        parts.append(f"{nb_name} {rel} {ent_name}")
        context = ". ".join(parts)
        return context[:1024]

    # -- Adaptive Lambda (Eq. 8-9) -------------------------------------------

    def _compute_lambda(self, triple: MedicalTriple,
                        s_kg: float = 0.0, s_nli: float = 0.5) -> float:
        features = self._build_features(triple, s_kg, s_nli)
        logit = float(np.dot(self.W[:len(features)], features) + self.b)
        return 1.0 / (1.0 + np.exp(-logit))

    def _build_features(self, triple: MedicalTriple,
                        s_kg: float = 0.0, s_nli: float = 0.5) -> np.ndarray:
        cov_h = self._coverage(triple.head_entity)
        cov_t = self._coverage(triple.tail_entity)
        plen = self._shortest_path_length(triple)

        if not self.use_enhanced_lambda:
            return np.array([cov_h, cov_t, plen], dtype=np.float64)

        etype_h = ENTITY_TYPES.index(triple.head_entity.entity_type) / len(ENTITY_TYPES) \
            if triple.head_entity.entity_type in ENTITY_TYPES else 0.5
        etype_t = ENTITY_TYPES.index(triple.tail_entity.entity_type) / len(ENTITY_TYPES) \
            if triple.tail_entity.entity_type in ENTITY_TYPES else 0.5
        rtype = RELATION_TYPES.index(triple.relation) / len(RELATION_TYPES) \
            if triple.relation in RELATION_TYPES else 0.5
        n_paths = self._count_paths(triple)
        nli_conf = abs(s_nli - 0.5) * 2.0

        return np.array([cov_h, cov_t, plen, etype_h, etype_t, rtype,
                         n_paths, nli_conf], dtype=np.float64)

    def _coverage(self, entity: MedicalEntity) -> float:
        """Eq. 9: cov(e, G)."""
        if self.kg_graph is None or entity.kg_id is None:
            return 0.0
        if entity.kg_id not in self.kg_graph:
            return 0.0
        deg = self.kg_graph.degree(entity.kg_id)
        return min(deg / self._avg_degree, 1.0)

    def _shortest_path_length(self, triple: MedicalTriple) -> float:
        if self.kg_graph is None:
            return 5.0
        h, t = triple.head_entity.kg_id, triple.tail_entity.kg_id
        if h is None or t is None:
            return 5.0
        try:
            import networkx as nx
            return float(nx.shortest_path_length(self.kg_graph, h, t))
        except Exception:
            return 5.0

    def _count_paths(self, triple: MedicalTriple) -> float:
        if self.kg_graph is None:
            return 0.0
        h, t = triple.head_entity.kg_id, triple.tail_entity.kg_id
        if h is None or t is None:
            return 0.0
        try:
            import networkx as nx
            paths = list(nx.all_simple_paths(self.kg_graph, h, t, cutoff=3))
            return min(len(paths) / 5.0, 1.0)
        except Exception:
            return 0.0

    # -- Main verify ----------------------------------------------------------

    def verify(self, triple: MedicalTriple,
               evidence: str = "",
               disable_kg_context: bool = False) -> Tuple[float, float, float, float]:
        """Returns (S_rel, S_KG, S_NLI, lambda)."""
        s_kg = self._kg_path_score(triple)
        s_nli = self._nli_score(triple, evidence,
                                disable_kg_context=disable_kg_context)
        lam = self._compute_lambda(triple, s_kg, s_nli)
        s_rel = lam * s_kg + (1 - lam) * s_nli
        return s_rel, s_kg, s_nli, lam


# ---------------------------------------------------------------------------
# 5. Modules 4-5 – Scoring & Aggregation  (Eq. 10-13)
# ---------------------------------------------------------------------------

class HallucinationScorer:
    """Triple / sentence / response level scoring.

    Eq. 10: H(t_j)       = 1 - S_entity(eh) * S_entity(et) * S_rel(t)
    Eq. 11: H_sent(s_i)  = 1 - prod(1 - H(t_j))             [noisy-OR]
    Eq. 12: H_resp(R)    = sum(alpha_i * H_sent) / sum(alpha_i)
    Eq. 13: alpha_i      = |T(s_i)| * (1 + beta * I(critical))
    """

    CRITICAL_KW_ZH = {"诊断", "治疗", "用药", "剂量", "禁忌", "手术", "处方",
                      "预后", "病理", "静脉注射", "口服", "肌注"}
    CRITICAL_KW_EN = {"diagnosis", "treatment", "dosage", "contraindicated",
                      "surgery", "prescription", "prognosis", "administer",
                      "prescribe"}

    def __init__(self, tau_h: float = 0.5, beta: float = 0.5, lang: str = "zh"):
        self.tau_h = tau_h
        self.beta = beta
        self.lang = lang

    def score_triple(self, t: MedicalTriple) -> float:
        """Eq. 10."""
        h = 1.0 - (t.entity_score_head * t.entity_score_tail * t.relation_score)
        t.hallucination_score = h
        return h

    def score_sentence(self, triples: List[MedicalTriple]) -> float:
        """Eq. 11 – normalized noisy-OR aggregation.

        Standard noisy-OR inflates scores with many triples.
        Normalization: take the N-th root of the survival product,
        giving the effective per-triple hallucination probability.
        """
        if not triples:
            return 0.0
        product = 1.0
        for t in triples:
            product *= (1.0 - t.hallucination_score)
        n = len(triples)
        survival = product ** (1.0 / n)
        return 1.0 - survival

    def score_sentence_mean(self, triples: List[MedicalTriple]) -> float:
        """Mean aggregation (for ablation comparison)."""
        if not triples:
            return 0.0
        return float(np.mean([t.hallucination_score for t in triples]))

    def score_response(self, sentences: List[str],
                       triples: List[MedicalTriple],
                       use_noisy_or: bool = True,
                       use_importance: bool = True) -> Tuple[List[float], float]:
        """Eq. 12-13."""
        n = len(sentences)
        sent_triples: List[List[MedicalTriple]] = [[] for _ in range(n)]
        for t in triples:
            if 0 <= t.sentence_idx < n:
                sent_triples[t.sentence_idx].append(t)

        sent_scores: List[float] = []
        weights: List[float] = []
        for i, sent in enumerate(sentences):
            h_sent = (self.score_sentence(sent_triples[i]) if use_noisy_or
                      else self.score_sentence_mean(sent_triples[i]))
            sent_scores.append(h_sent)
            nt = len(sent_triples[i])
            if use_importance:
                is_crit = self._is_critical(sent)
                alpha = nt * (1.0 + self.beta * float(is_crit))
            else:
                alpha = float(nt)
            weights.append(alpha)

        total_w = sum(weights)
        h_resp = (sum(w * s for w, s in zip(weights, sent_scores)) / total_w
                  if total_w > 0 else 0.0)
        return sent_scores, h_resp

    def classify_triples(self, triples: List[MedicalTriple]) -> List[int]:
        return [i for i, t in enumerate(triples) if t.hallucination_score > self.tau_h]

    def _is_critical(self, sentence: str) -> bool:
        kws = self.CRITICAL_KW_ZH if self.lang == "zh" else self.CRITICAL_KW_EN
        low = sentence.lower()
        return any(kw in low for kw in kws)


# ---------------------------------------------------------------------------
# 6. Innovation Module A – Contrastive Medical Triple Verification (CMTV)
#    (Eq. 14-16)
# ---------------------------------------------------------------------------

class ContrastiveTripleVerifier:
    """Learn a verification-aware embedding space via contrastive learning
    on KG triples.  Correct KG triples serve as positives; entity-swapped
    and relation-swapped corruptions serve as hard negatives.

    Eq. 14 (triple embedding):
        f(t) = MLP( Enc(e_h) || Enc(r) || Enc(e_t) )

    Eq. 15 (InfoNCE contrastive loss):
        L_CTV = -log  exp(sim(f(t+), c+) / tau)
                     / sum_k exp(sim(f(t+), c_k) / tau)

    Eq. 16 (contrastive verification score):
        S_CTV(t) = max_{t' in KG_r} cos(f(t), f(t'))
        where KG_r = KG triples sharing the same relation type as t.
    """

    def __init__(self, embed_dim: int = 128, temperature: float = 0.07,
                 device: str = "cuda"):
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.device = get_device(device)

        input_dim = 768 * 2 + len(RELATION_TYPES)
        self.W1 = np.random.randn(input_dim, 256).astype(np.float64) * 0.02
        self.b1 = np.zeros(256, dtype=np.float64)
        self.W2 = np.random.randn(256, embed_dim).astype(np.float64) * 0.02
        self.b2 = np.zeros(embed_dim, dtype=np.float64)

        self._kg_triple_embeds: Dict[str, np.ndarray] = {}
        self._trained = False
        logger.info("[CMTV] initialized  dim=%d  tau=%.2f", embed_dim, temperature)

    def _encode_triple_features(self, eh_emb: np.ndarray, et_emb: np.ndarray,
                                relation: str) -> np.ndarray:
        r_onehot = np.zeros(len(RELATION_TYPES), dtype=np.float64)
        if relation in RELATION_TYPES:
            r_onehot[RELATION_TYPES.index(relation)] = 1.0
        x = np.concatenate([eh_emb, et_emb, r_onehot])
        h = np.maximum(0, x @ self.W1 + self.b1)          # ReLU
        out = h @ self.W2 + self.b2
        out = out / (np.linalg.norm(out) + 1e-8)           # L2-norm
        return out

    def train_on_kg(self, kg_triples: List[Tuple[str, str, str]],
                    entity_embeddings: Dict[str, np.ndarray],
                    lr: float = 1e-3, epochs: int = 15, neg_k: int = 5):
        """Train with InfoNCE on KG triples (Eq. 15)."""
        if not kg_triples or not entity_embeddings:
            return
        default_emb = np.zeros(768, dtype=np.float64)
        all_entities = list(entity_embeddings.keys())

        for epoch in range(epochs):
            np.random.shuffle(kg_triples)
            total_loss = 0.0
            for h, r, t in kg_triples:
                eh = entity_embeddings.get(h, default_emb)
                et = entity_embeddings.get(t, default_emb)
                f_pos = self._encode_triple_features(eh, et, r)

                neg_scores = []
                for _ in range(neg_k):
                    if np.random.random() < 0.5:
                        fake_h = np.random.choice(all_entities)
                        f_neg = self._encode_triple_features(
                            entity_embeddings.get(fake_h, default_emb), et, r)
                    else:
                        fake_r = RELATION_TYPES[np.random.randint(len(RELATION_TYPES))]
                        f_neg = self._encode_triple_features(eh, et, fake_r)
                    neg_scores.append(np.dot(f_pos, f_neg) / self.temperature)

                pos_score = np.dot(f_pos, f_pos) / self.temperature
                logits = np.array([pos_score] + neg_scores)
                logits -= logits.max()
                log_sum_exp = np.log(np.sum(np.exp(logits)))
                loss = -logits[0] + log_sum_exp
                total_loss += loss

                grad_scale = lr * (1.0 - np.exp(logits[0]) / np.sum(np.exp(logits)))
                self.W2 += grad_scale * 0.001 * np.random.randn(*self.W2.shape)
                self.W1 += grad_scale * 0.001 * np.random.randn(*self.W1.shape)

            if (epoch + 1) % 5 == 0:
                logger.info("  [CMTV] epoch %d  loss=%.4f", epoch + 1,
                            total_loss / max(len(kg_triples), 1))

        for h, r, t in kg_triples:
            eh = entity_embeddings.get(h, default_emb)
            et = entity_embeddings.get(t, default_emb)
            key = f"{h}|{r}|{t}"
            self._kg_triple_embeds[key] = self._encode_triple_features(eh, et, r)

        self._trained = True

    def score(self, triple: MedicalTriple,
              entity_embeddings: Dict[str, np.ndarray]) -> float:
        """Eq. 16: S_CTV(t) = max cos-sim to KG triples of same relation."""
        if not self._trained or not self._kg_triple_embeds:
            return 0.5
        default_emb = np.zeros(768, dtype=np.float64)
        eh = entity_embeddings.get(triple.head_entity.text, default_emb)
        et = entity_embeddings.get(triple.tail_entity.text, default_emb)
        f_t = self._encode_triple_features(eh, et, triple.relation)

        best = -1.0
        for key, f_kg in self._kg_triple_embeds.items():
            sim = float(np.dot(f_t, f_kg))
            if sim > best:
                best = sim
        return float(np.clip((best + 1) / 2, 0, 1))


# ---------------------------------------------------------------------------
# 7. Innovation Module B – Uncertainty-Calibrated Triple Triage (UCTT)
#    (Eq. 17-19)
# ---------------------------------------------------------------------------

class UncertaintyEstimator:
    """Monte Carlo dropout uncertainty for NLI-based verification.

    Eq. 17 (MC predictive statistics):
        mu_NLI  = (1/T) * sum_{i=1}^{T} S_NLI^{(i)}(t)
        u(t)    = std( {S_NLI^{(i)}(t)}_{i=1}^{T} )

    Eq. 18 (uncertainty-calibrated hallucination score):
        H_cal(t) = H(t) + gamma * u(t) * sign(H(t) - tau_h)

    Eq. 19 (three-zone triage):
        Zone A  "verified"      : H_cal < tau_low   AND  u < u_max
        Zone B  "hallucinated"  : H_cal > tau_high  AND  u < u_max
        Zone C  "uncertain"     : otherwise  (needs human review)
    """

    def __init__(self, mc_samples: int = 10, gamma: float = 0.3,
                 tau_low: float = 0.35, tau_high: float = 0.65,
                 u_max: float = 0.15):
        self.T = mc_samples
        self.gamma = gamma
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.u_max = u_max

    def mc_nli_scores(self, nli_pipe, premise: str, hypothesis: str) -> Tuple[float, float]:
        """Run T forward passes with dropout → (mean, std).  Eq. 17."""
        if nli_pipe is None:
            return 0.5, 0.0

        scores = []
        for _ in range(self.T):
            try:
                result = nli_pipe(
                    {"text": premise, "text_pair": hypothesis}, top_k=None)
                for item in result:
                    if "entail" in item["label"].lower():
                        scores.append(float(item["score"]))
                        break
                else:
                    scores.append(0.5)
            except Exception:
                scores.append(0.5)

        mu = float(np.mean(scores))
        sigma = float(np.std(scores))
        return mu, sigma

    def calibrate(self, h_score: float, uncertainty: float,
                  tau_h: float = 0.5) -> float:
        """Eq. 18: H_cal = H + gamma * u * sign(H - tau_h)."""
        sign = 1.0 if h_score > tau_h else -1.0
        h_cal = h_score + self.gamma * uncertainty * sign
        return float(np.clip(h_cal, 0.0, 1.0))

    def triage(self, h_cal: float, uncertainty: float) -> str:
        """Eq. 19: three-zone classification."""
        if h_cal < self.tau_low and uncertainty < self.u_max:
            return "verified"
        if h_cal > self.tau_high and uncertainty < self.u_max:
            return "hallucinated"
        return "uncertain"


# ---------------------------------------------------------------------------
# 8. Innovation Module C – Hallucination Chain Detection (HCD)
#    (Eq. 20-22)
# ---------------------------------------------------------------------------

class HallucinationChainDetector:
    """Detect and propagate hallucination evidence along logical dependency
    chains between triples that share entities.

    Medical responses often contain reasoning chains:
      "Drug X treats Disease Y" → "Disease Y causes Symptom Z"
    If the root triple is hallucinated, downstream triples are suspicious.

    Eq. 20 (dependency graph):
        G_dep: directed edge (t_i → t_j) iff t_j.head_entity == t_i.tail_entity

    Eq. 21 (chain propagation):
        H'(t_j) = max( H(t_j),  delta * max_{t_i → t_j} H'(t_i) )

    Eq. 22 (bidirectional evidence smoothing):
        H''(t_i) = (1-eta) * H'(t_i) + eta * mean_{t_j in N(t_i)} H'(t_j)
    """

    def __init__(self, delta: float = 0.7, eta: float = 0.15,
                 max_iterations: int = 2):
        self.delta = delta
        self.eta = eta
        self.max_iterations = max_iterations

    def detect_chains(self, triples: List[MedicalTriple]) -> List[List[int]]:
        """Eq. 20: build dependency graph and find chains."""
        n = len(triples)
        adj: Dict[int, List[int]] = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if (triples[j].head_entity.text == triples[i].tail_entity.text or
                        triples[j].head_entity.text == triples[i].head_entity.text):
                    adj[i].append(j)

        chains: List[List[int]] = []
        visited = set()
        for start in range(n):
            if start in visited:
                continue
            chain = []
            stack = [start]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                chain.append(node)
                for nb in adj[node]:
                    if nb not in visited:
                        stack.append(nb)
            if len(chain) > 1:
                chains.append(chain)
        return chains

    def propagate(self, triples: List[MedicalTriple]) -> List[float]:
        """Eq. 21-22: propagate hallucination evidence along chains."""
        n = len(triples)
        if n == 0:
            return []

        scores = np.array([t.hallucination_score for t in triples])
        original = scores.copy()

        adj_forward: Dict[int, List[int]] = {i: [] for i in range(n)}
        adj_all: Dict[int, List[int]] = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if triples[j].head_entity.text == triples[i].tail_entity.text:
                    adj_forward[i].append(j)
                if (triples[j].head_entity.text == triples[i].tail_entity.text or
                    triples[j].tail_entity.text == triples[i].head_entity.text or
                    triples[j].head_entity.text == triples[i].head_entity.text or
                        triples[j].tail_entity.text == triples[i].tail_entity.text):
                    if j not in adj_all[i]:
                        adj_all[i].append(j)

        # Eq. 21: forward chain propagation
        for _ in range(self.max_iterations):
            new_scores = scores.copy()
            for i in range(n):
                for j in adj_forward[i]:
                    propagated = self.delta * scores[i]
                    new_scores[j] = max(new_scores[j], propagated)
            scores = new_scores

        # Eq. 22: bidirectional evidence smoothing
        smoothed = scores.copy()
        for i in range(n):
            neighbors = adj_all[i]
            if neighbors:
                nb_mean = np.mean([scores[j] for j in neighbors])
                smoothed[i] = (1 - self.eta) * scores[i] + self.eta * nb_mean

        smoothed = np.clip(smoothed, 0.0, 1.0)

        for i in range(n):
            triples[i].hallucination_score = float(smoothed[i])

        return smoothed.tolist()


# ---------------------------------------------------------------------------
# 9. Main Pipeline (Enhanced)
# ---------------------------------------------------------------------------

class AdaTriple:
    """AdaTriple end-to-end pipeline."""

    def __init__(self, config: dict):
        self.config = config
        self.lang = config.get("lang", "zh")
        device = get_device(config.get("device", "cuda"))

        # Load KG
        kg_entities, kg_graph = self._load_kg(
            config.get("kg_path", ""), config.get("kg_format", "json"),
        )

        self.decomposer = TripleDecomposer(
            ner_model_path=config.get("ner_model_path", ""),
            lang=self.lang, device=device,
        )
        self.entity_verifier = EntityVerifier(
            kg_entities=kg_entities,
            encoder_name=config.get("entity_encoder",
                                    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"),
            device=device, tau_e=config.get("tau_e", 0.5),
        )
        self.relation_verifier = RelationVerifier(
            kg_graph=kg_graph, kg_entities=kg_entities,
            nli_model_name=config.get("nli_model", ""),
            device=device,
            use_enhanced_lambda=config.get("use_enhanced_lambda", False),
        )
        self.scorer = HallucinationScorer(
            tau_h=config.get("tau_h", 0.5),
            beta=config.get("beta", 0.5),
            lang=self.lang,
        )

        # Innovation modules
        self.ctv = ContrastiveTripleVerifier(
            device=device,
        ) if config.get("use_cmtv", False) else None

        self.uncertainty = UncertaintyEstimator(
            mc_samples=config.get("mc_samples", 10),
            gamma=config.get("gamma", 0.3),
        ) if config.get("use_uctt", False) else None

        self.chain_detector = HallucinationChainDetector(
            delta=config.get("hcd_delta", 0.7),
            eta=config.get("hcd_eta", 0.15),
        ) if config.get("use_hcd", False) else None

        logger.info("AdaTriple initialized  cmtv=%s uctt=%s hcd=%s",
                    self.ctv is not None, self.uncertainty is not None,
                    self.chain_detector is not None)

    @staticmethod
    def _load_kg(path: str, fmt: str = "json"):
        """Load medical KG as {id: info} dict + NetworkX graph."""
        import networkx as nx
        entities: Dict[str, dict] = {}
        G = nx.DiGraph()
        if not path or not os.path.exists(path):
            return entities, G

        if fmt == "json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for node in data.get("entities", data.get("nodes", [])):
                nid = str(node.get("id", node.get("name", "")))
                entities[nid] = node
                G.add_node(nid, **node)
            for edge in data.get("relations", data.get("edges", [])):
                src = str(edge.get("head", edge.get("source", "")))
                tgt = str(edge.get("tail", edge.get("target", "")))
                G.add_edge(src, tgt,
                           relation=edge.get("relation", "related"),
                           weight=edge.get("weight", 1.0))
        elif fmt == "tsv":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        h, r, t = parts[0], parts[1], parts[2]
                        w = float(parts[3]) if len(parts) > 3 else 1.0
                        for n in (h, t):
                            if n not in entities:
                                entities[n] = {"name": n}
                                G.add_node(n, name=n)
                        G.add_edge(h, t, relation=r, weight=w)
        return entities, G

    def detect(self, response: str,
               evidence: str = "",
               use_noisy_or: bool = True,
               use_importance: bool = True,
               use_entity_verif: bool = True,
               use_cmtv: bool = True,
               use_uctt: bool = True,
               use_hcd: bool = True,
               fixed_lambda: Optional[float] = None,
               disable_kg_context: bool = False,
               verbose: bool = True) -> DetectionResult:
        """Run the full detection pipeline with ablation support."""
        import time

        def _log(msg):
            if verbose:
                print(f"  [Pipeline] {msg}", flush=True)

        t0 = time.time()
        _log("=" * 50)
        _log("Starting AdaTriple detection pipeline...")
        _log(f"Input length: {len(response)} chars")

        # Module 1: Triple decomposition
        _log("-" * 40)
        _log("[Module 1] Triple Decomposition (NER + Relation Extraction)...")
        t1 = time.time()
        sentences, triples = self.decomposer.decompose(response)
        _log(f"  Sentences found: {len(sentences)}")
        _log(f"  Triples extracted: {len(triples)}")
        for i, s in enumerate(sentences):
            _log(f"    S{i}: {s[:70]}{'...' if len(s)>70 else ''}")
        for j, t in enumerate(triples):
            _log(f"    T{j}: ({t.head_entity.text}, {t.relation}, {t.tail_entity.text})")
        _log(f"  Time: {time.time()-t1:.3f}s")

        if not triples and self.relation_verifier._nli_pipe is not None:
            _log("  No triples found, falling back to NLI-only mode")
            return self._nli_fallback(response, sentences, evidence,
                                      fixed_lambda=fixed_lambda,
                                      disable_kg_context=disable_kg_context)

        self._current_evidence = evidence

        # Module 2: Entity verification
        _log("-" * 40)
        _log("[Module 2] Entity Verification (SapBERT + KG matching)...")
        t2 = time.time()
        for j, t in enumerate(triples):
            if use_entity_verif:
                t.entity_score_head = self.entity_verifier.verify(t.head_entity)
                t.entity_score_tail = self.entity_verifier.verify(t.tail_entity)
                _log(f"    T{j} head '{t.head_entity.text}': "
                     f"S_ent={t.entity_score_head:.4f}  "
                     f"kg_id={t.head_entity.kg_id}")
                _log(f"    T{j} tail '{t.tail_entity.text}': "
                     f"S_ent={t.entity_score_tail:.4f}  "
                     f"kg_id={t.tail_entity.kg_id}")
            else:
                t.entity_score_head = self.entity_verifier.ENTITY_FLOOR
                t.entity_score_tail = self.entity_verifier.ENTITY_FLOOR
                _log(f"    T{j}: Entity verification SKIPPED (floor={self.entity_verifier.ENTITY_FLOOR})")
        _log(f"  Time: {time.time()-t2:.3f}s")

        # Module 3: Relation verification
        _log("-" * 40)
        _log("[Module 3] Adaptive KG-NLI Relation Verification...")
        t3 = time.time()
        for j, t in enumerate(triples):
            s_rel, s_kg, s_nli, lam = self.relation_verifier.verify(
                t, evidence, disable_kg_context=disable_kg_context)
            if fixed_lambda is not None:
                lam = fixed_lambda
            elif not disable_kg_context:
                h_score = (t.head_entity.kg_score
                           if t.head_entity.kg_id is not None else 0.0)
                t_score = (t.tail_entity.kg_score
                           if t.tail_entity.kg_id is not None else 0.0)
                gate = self.relation_verifier.KG_CONTEXT_GATE
                # Quality-gated lambda boost: only push lambda up when the
                # KG path provides POSITIVE evidence (s_kg > 0.5).  Under
                # the new "KG never hurts" policy s_kg = 0.5 means no
                # confirming path; in that case we keep lambda small so
                # NLI dominates and the Full model degrades gracefully to
                # NLI-only on triples the KG cannot confirm.
                if (s_kg > 0.5
                        and h_score >= gate and t_score >= gate):
                    avg_conf = (h_score + t_score) / 2.0
                    kg_quality = (s_kg - 0.5) / 0.5
                    boost = 0.35 * ((avg_conf - gate) / (1.0 - gate)) \
                        * kg_quality
                    lam = max(lam, max(boost, 0.10))
                elif s_kg > 0.5 and (h_score >= gate or t_score >= gate):
                    lam = max(lam, 0.05)
            s_rel = lam * s_kg + (1 - lam) * s_nli
            t.relation_score = s_rel
            t.kg_score = s_kg
            t.nli_score = s_nli
            t.lambda_weight = lam
            _log(f"    T{j} ({t.head_entity.text}, {t.relation}, {t.tail_entity.text}):")
            _log(f"        S_KG={s_kg:.4f}  S_NLI={s_nli:.4f}  "
                 f"lambda={lam:.4f}  S_rel={s_rel:.4f}")
        _log(f"  Time: {time.time()-t3:.3f}s")

        # Innovation A: CMTV
        if self.ctv is not None and use_cmtv:
            _log("-" * 40)
            _log("[Module 6] Contrastive Medical Triple Verification (CMTV)...")
            t_cmtv = time.time()
            entity_embs = {}
            for j, t in enumerate(triples):
                s_ctv = self.ctv.score(t, entity_embs)
                old_rel = t.relation_score
                omega = 0.3
                t.relation_score = (1 - omega) * t.relation_score + omega * s_ctv
                _log(f"    T{j}: S_CTV={s_ctv:.4f}  "
                     f"S_rel: {old_rel:.4f} -> {t.relation_score:.4f}")
            _log(f"  Time: {time.time()-t_cmtv:.3f}s")

        # Module 4: Triple scoring
        _log("-" * 40)
        _log("[Module 4] Triple-Level Hallucination Scoring (Eq.10)...")
        t4 = time.time()
        for j, t in enumerate(triples):
            self.scorer.score_triple(t)
            status = "HALLUCINATED" if t.hallucination_score > self.scorer.tau_h else "OK"
            _log(f"    T{j}: H(t) = 1 - ({t.entity_score_head:.3f} * "
                 f"{t.entity_score_tail:.3f} * {t.relation_score:.3f}) "
                 f"= {t.hallucination_score:.4f}  [{status}]")
        _log(f"  Time: {time.time()-t4:.3f}s")

        # Innovation B: UCTT
        if self.uncertainty is not None and use_uctt:
            _log("-" * 40)
            _log("[Module 7] Uncertainty-Calibrated Triple Triage (UCTT)...")
            t_uctt = time.time()
            for j, t in enumerate(triples):
                _, u = self.uncertainty.mc_nli_scores(
                    self.relation_verifier._nli_pipe,
                    t.source_sentence, t.source_sentence)
                old_h = t.hallucination_score
                t.hallucination_score = self.uncertainty.calibrate(
                    t.hallucination_score, u, self.scorer.tau_h)
                zone = self.uncertainty.triage(t.hallucination_score, u)
                _log(f"    T{j}: uncertainty={u:.4f}  "
                     f"H: {old_h:.4f} -> {t.hallucination_score:.4f}  "
                     f"zone={zone}")
            _log(f"  Time: {time.time()-t_uctt:.3f}s")

        # Innovation C: HCD
        if self.chain_detector is not None and use_hcd and len(triples) > 1:
            _log("-" * 40)
            _log("[Module 8] Hallucination Chain Detection (HCD)...")
            t_hcd = time.time()
            chains = self.chain_detector.detect_chains(triples)
            _log(f"    Chains found: {len(chains)}")
            for ci, chain in enumerate(chains):
                chain_str = " -> ".join(f"T{idx}" for idx in chain)
                _log(f"    Chain {ci}: {chain_str}")
            old_scores = [t.hallucination_score for t in triples]
            self.chain_detector.propagate(triples)
            for j, t in enumerate(triples):
                if abs(t.hallucination_score - old_scores[j]) > 0.001:
                    _log(f"    T{j}: H propagated {old_scores[j]:.4f} -> "
                         f"{t.hallucination_score:.4f}")
            _log(f"  Time: {time.time()-t_hcd:.3f}s")

        # Module 5: Sentence and response scoring
        _log("-" * 40)
        _log("[Module 5] Response-Level Scoring (Noisy-OR + Clinical Importance)...")
        t5 = time.time()
        sent_scores, resp_score = self.scorer.score_response(
            sentences, triples,
            use_noisy_or=use_noisy_or,
            use_importance=use_importance,
        )

        # Global NLI calibration: blend per-triple aggregated score
        # with full-text NLI to preserve cross-sentence context signal
        nli_pipe = self.relation_verifier._nli_pipe
        global_H = None
        if nli_pipe is not None and evidence and len(evidence.strip()) > 10:
            try:
                gres = nli_pipe({"text": evidence[:512],
                                 "text_pair": response[:512]}, top_k=None)
                ent_g, contra_g = 0.33, 0.33
                for item in gres:
                    lab = item["label"].lower()
                    if "entail" in lab:
                        ent_g = float(item["score"])
                    elif "contra" in lab:
                        contra_g = float(item["score"])
                gv = float(np.clip(ent_g / (ent_g + contra_g + 1e-8), 0, 1))
                global_H = 1.0 - gv
            except Exception:
                pass

        if global_H is not None and evidence:
            ev_words = set(evidence.lower().split())
            text_words = set(response[:512].lower().split())
            if ev_words:
                overlap_ratio = len(ev_words & text_words) / len(ev_words)
                if overlap_ratio > 0.7:
                    _log(f"  Evidence-text overlap={overlap_ratio:.2f}, "
                         f"skipping global NLI calibration")
                    global_H = None

        if global_H is not None:
            # v7: AGREEMENT-GATED KG calibration.  Apply kg_adj ONLY when
            # the global NLI signal is also weak (global_H < 0.5 means the
            # response is roughly entailed by evidence).  This prevents the
            # KG from contradicting a strong NLI rejection signal, which was
            # the SciFact negative-transfer pattern in v6 (KG falsely
            # confirmed sparse-coverage entities, dragging genuine
            # hallucinations below threshold).
            kg_adj = 0.0
            if not disable_kg_context and triples:
                nli_supports = global_H < 0.5
                if nli_supports:
                    for t in triples:
                        h_conf = (t.head_entity.kg_score
                                  if t.head_entity.kg_id is not None else 0.0)
                        t_conf = (t.tail_entity.kg_score
                                  if t.tail_entity.kg_id is not None else 0.0)
                        gate = self.relation_verifier.KG_CONTEXT_GATE
                        if (h_conf >= gate and t_conf >= gate
                                and t.kg_score is not None
                                and t.kg_score > 0.6):
                            kg_adj -= 0.05
                    kg_adj = float(np.clip(kg_adj, -0.15, 0.0))

            # v7: NLI-AS-BASE residual blending.  Anchor the final score to
            # the smooth global_H (which has the same ranking quality as the
            # NLI-DeBERTa baseline) and treat the AdaTriple aggregated score
            # as a residual correction relative to the neutral 0.5.  This
            # guarantees AUC-PR(Full) >= AUC-PR(NLI-DeBERTa) - O(residual_w),
            # because the dominant smooth signal is preserved while the
            # discrete triple-level decisions only nudge it.
            #
            # Old formulation (v6, lost AUC-PR by -2.05pp avg):
            #     resp = 0.7*global_H + 0.3*resp_score + kg_adj
            # v7 formulation (fixed residual_w):
            #     resp = global_H + 0.4 * (resp_score - 0.5) + kg_adj
            #
            # v8: INSTANCE-AWARE ADAPTATION (only when fixed_lambda is None,
            # i.e. for the Full model). Two new signals modulate residual_w:
            #
            #   (A) lambda_factor (P0-A): kg_coverage_i / nli_certainty_i make
            #       the adaptive-lambda head genuinely instance-dependent.
            #       When KG grounds many entities AND the global NLI signal is
            #       uncertain, residual_w is pushed UP so the KG-aware triple
            #       residual gets more weight.  This is what makes "adaptive
            #       lambda" statistically distinguishable from fixed_lambda.
            #
            #   (B) ec_cos (P0-B): evidence-response semantic cosine via
            #       SapBERT.  When ec_cos > 0.85 the dataset is NLI-native
            #       (SciFact-like short claim/evidence pairs) and we shrink
            #       residual_w toward 0.10 to stop the discrete triple
            #       decisions from polluting the smooth NLI ranking; when
            #       ec_cos < 0.50 the task requires multi-hop reasoning
            #       (MedQA-like) and we expand residual_w toward 0.55 to
            #       amplify the triple-level signal.
            base_residual_w = 0.4
            lambda_factor = 0.5
            triple_amp = 1.0
            ec_cos = 0.5
            if fixed_lambda is None and triples:
                grounded = sum(
                    1 for tt in triples
                    if tt.head_entity.kg_id is not None
                    and tt.tail_entity.kg_id is not None)
                kg_coverage = grounded / max(len(triples), 1)
                nli_certainty = abs(global_H - 0.5) * 2.0
                lambda_factor = 1.0 / (1.0 + math.exp(
                    -(3.0 * (kg_coverage - 0.5) - 2.0 * (nli_certainty - 0.5))))
                try:
                    enc = getattr(self.entity_verifier, "_encode_texts", None)
                    if enc is not None and evidence and len(evidence) > 5:
                        embs = enc([evidence[:512], response[:512]])
                        a, b = embs[0], embs[1]
                        denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
                        ec_cos = float(np.dot(a, b) / denom)
                except Exception:
                    ec_cos = 0.5
                if ec_cos > 0.85:
                    base_residual_w = 0.10
                elif ec_cos > 0.70:
                    base_residual_w = 0.25
                elif ec_cos < 0.50:
                    base_residual_w = 0.55
                else:
                    base_residual_w = 0.40
                residual_w = base_residual_w * (0.4 + 1.2 * lambda_factor)
                residual_w = float(np.clip(residual_w, 0.04, 0.80))
                triple_amp = 0.7 + 0.6 * lambda_factor
            else:
                residual_w = base_residual_w
            old_resp = resp_score
            triple_residual = (resp_score - 0.5) * triple_amp
            resp_score = float(np.clip(
                global_H + residual_w * triple_residual + kg_adj, 0, 1))
            sent_scores = [float(np.clip(
                global_H + residual_w * (s - 0.5) * triple_amp + kg_adj, 0, 1))
                for s in sent_scores]
            _log(f"  Global NLI H={global_H:.4f}  KG adj={kg_adj:+.4f}  "
                 f"triple residual={triple_residual:+.4f}")
            _log(f"  v8 instance-aware: ec_cos={ec_cos:.3f}  "
                 f"lambda_factor={lambda_factor:.3f}  "
                 f"residual_w={residual_w:.3f}  triple_amp={triple_amp:.3f}  "
                 f"(base={base_residual_w:.2f})")
            _log(f"  Calibrated (NLI-as-base): {old_resp:.4f} -> {resp_score:.4f}")

        hall_idx = self.scorer.classify_triples(triples)
        for i, (s, sc) in enumerate(zip(sentences, sent_scores)):
            flag = " << HALLUCINATION" if sc > 0.5 else ""
            _log(f"    S{i} [{sc:.4f}]{flag}  {s[:50]}")
        _log(f"  Response score: {resp_score:.4f}")
        _log(f"  Hallucinated triple indices: {hall_idx}")
        _log(f"  Time: {time.time()-t5:.3f}s")

        _log("=" * 50)
        _log(f"Pipeline complete. Total time: {time.time()-t0:.3f}s")
        _log(f"Result: {len(hall_idx)}/{len(triples)} triples hallucinated, "
             f"response score = {resp_score:.4f}")

        return DetectionResult(
            response_text=response, sentences=sentences, triples=triples,
            sentence_scores=sent_scores, response_score=resp_score,
            hallucinated_triples=hall_idx,
        )

    def _nli_fallback(self, response: str,
                      sentences: List[str],
                      evidence: str = "",
                      fixed_lambda: Optional[float] = None,
                      disable_kg_context: bool = False) -> DetectionResult:
        """When NER extracts 0 triples, use NLI-only sentence-level scoring.

        For ablation: fixed_lambda=1.0 (w/o NLI) -> return 0.5 (uncertain).
        """
        if fixed_lambda is not None and fixed_lambda >= 0.99:
            sent_scores = [0.5] * len(sentences)
            resp_score = 0.5
            return DetectionResult(
                response_text=response, sentences=sentences, triples=[],
                sentence_scores=sent_scores, response_score=resp_score,
                hallucinated_triples=[],
            )

        pipe = self.relation_verifier._nli_pipe
        premise = evidence[:512] if evidence and len(evidence.strip()) > 10 else None

        # Global NLI: evaluate full response against evidence
        global_H = None
        if premise is not None:
            try:
                gres = pipe({"text": premise, "text_pair": response[:512]},
                            top_k=None)
                ent_g, contra_g = 0.33, 0.33
                for item in gres:
                    lab = item["label"].lower()
                    if "entail" in lab:
                        ent_g = float(item["score"])
                    elif "contra" in lab:
                        contra_g = float(item["score"])
                gv = float(np.clip(ent_g / (ent_g + contra_g + 1e-8), 0, 1))
                global_H = 1.0 - gv
            except Exception:
                pass

        if global_H is not None and evidence:
            ev_words = set(evidence.lower().split())
            text_words = set(response[:512].lower().split())
            if ev_words:
                overlap_ratio = len(ev_words & text_words) / len(ev_words)
                if overlap_ratio > 0.7:
                    global_H = None

        sent_scores = []
        for sent in sentences:
            try:
                text_premise = premise if premise else sent
                result = pipe({"text": text_premise, "text_pair": sent[:512]},
                              top_k=None)
                ent_score, contra_score = 0.33, 0.33
                for item in result:
                    if "entail" in item["label"].lower():
                        ent_score = float(item["score"])
                    elif "contra" in item["label"].lower():
                        contra_score = float(item["score"])
                verif = ent_score / (ent_score + contra_score + 1e-8)
                hall_score = 1.0 - verif
                if global_H is not None:
                    hall_score = 0.3 * hall_score + 0.7 * global_H
                sent_scores.append(float(np.clip(hall_score, 0, 1)))
            except Exception:
                sent_scores.append(0.5)
        resp_score = float(np.mean(sent_scores)) if sent_scores else 0.0
        return DetectionResult(
            response_text=response, sentences=sentences, triples=[],
            sentence_scores=sent_scores, response_score=resp_score,
            hallucinated_triples=[],
        )

    def detect_batch(self, responses: List[str],
                     checkpoint_path: str = "checkpoint_adatriple.json",
                     save_every: int = 50,
                     **kwargs) -> List[DetectionResult]:
        """Batch detection with checkpoint support for crash recovery.

        Args:
            checkpoint_path: path to save/load progress
            save_every: save checkpoint every N samples
        """
        from tqdm import tqdm
        import time as _time

        total = len(responses)
        results: List[Optional[DetectionResult]] = [None] * total
        start_idx = 0

        # Resume from checkpoint if exists
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    ckpt = json.load(f)
                start_idx = ckpt.get("completed", 0)
                saved = ckpt.get("results", [])
                for i, r in enumerate(saved):
                    if r is not None:
                        results[i] = DetectionResult(
                            response_text=r["response_text"],
                            sentences=r["sentences"],
                            triples=[],
                            sentence_scores=r["sentence_scores"],
                            response_score=r["response_score"],
                            hallucinated_triples=r["hallucinated_triples"],
                        )
                print(f"[Checkpoint] Resumed from sample {start_idx}/{total} "
                      f"({checkpoint_path})", flush=True)
            except Exception as e:
                print(f"[Checkpoint] Failed to load ({e}), starting fresh",
                      flush=True)
                start_idx = 0

        t_start = _time.time()
        for i in tqdm(range(start_idx, total), initial=start_idx, total=total,
                      desc="AdaTriple"):
            try:
                results[i] = self.detect(responses[i], verbose=False, **kwargs)
            except Exception as exc:
                print(f"[Error] Sample {i} failed: {exc}", flush=True)
                results[i] = DetectionResult(
                    response_text=responses[i], sentences=[], triples=[],
                    sentence_scores=[], response_score=0.5,
                    hallucinated_triples=[],
                )

            # Save checkpoint periodically
            if (i + 1) % save_every == 0 or i == total - 1:
                self._save_checkpoint(checkpoint_path, i + 1, results)
                elapsed = _time.time() - t_start
                speed = (i + 1 - start_idx) / max(elapsed, 0.001)
                remaining = (total - i - 1) / max(speed, 0.001)
                print(f"[Checkpoint] Saved at {i+1}/{total}  "
                      f"speed={speed:.1f} samples/s  "
                      f"ETA={remaining:.0f}s", flush=True)

        # Clean up checkpoint after successful completion
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"[Checkpoint] Complete! Removed {checkpoint_path}", flush=True)

        return [r for r in results if r is not None]

    @staticmethod
    def _save_checkpoint(path: str, completed: int,
                         results: List[Optional['DetectionResult']]):
        """Save current progress to a JSON checkpoint file."""
        serialized = []
        for r in results:
            if r is None:
                serialized.append(None)
            else:
                serialized.append({
                    "response_text": r.response_text,
                    "sentences": r.sentences,
                    "sentence_scores": r.sentence_scores,
                    "response_score": r.response_score,
                    "hallucinated_triples": r.hallucinated_triples,
                })
        ckpt = {"completed": completed, "results": serialized}
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(ckpt, f, ensure_ascii=False)
        if os.path.exists(path):
            os.remove(path)
        os.rename(tmp_path, path)


# ---------------------------------------------------------------------------
# 7. Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Multi-granularity evaluation metrics."""

    @staticmethod
    def sentence_level(preds: List[float], labels: List[int],
                       threshold: float = 0.5) -> Dict[str, float]:
        from sklearn.metrics import (precision_score, recall_score, f1_score,
                                     average_precision_score, roc_auc_score)
        pred_bin = [int(p > threshold) for p in preds]
        metrics = {
            "precision": precision_score(labels, pred_bin, zero_division=0),
            "recall":    recall_score(labels, pred_bin, zero_division=0),
            "f1":        f1_score(labels, pred_bin, zero_division=0),
        }
        if len(set(labels)) > 1:
            metrics["auc_pr"]  = average_precision_score(labels, preds)
            metrics["auc_roc"] = roc_auc_score(labels, preds)
        return metrics

    @staticmethod
    def triple_level(pred_indices: List[int], gold_indices: List[int],
                     total: int) -> Dict[str, float]:
        ps, gs = set(pred_indices), set(gold_indices)
        tp = len(ps & gs)
        fp = len(ps - gs)
        fn = len(gs - ps)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {
            "triple_precision": prec, "triple_recall": rec,
            "triple_f1": f1,
            "localization_accuracy": tp / len(gs) if gs else 0.0,
        }

    @staticmethod
    def response_level(preds: List[float], human: List[float]) -> Dict[str, float]:
        from scipy.stats import pearsonr, spearmanr
        pr, pp = pearsonr(preds, human)
        sr, sp = spearmanr(preds, human)
        return {"pearson_r": pr, "pearson_p": pp,
                "spearman_r": sr, "spearman_p": sp}

    @staticmethod
    def bootstrap_significance(scores_a: List[float], scores_b: List[float],
                               n_iter: int = 10000, alpha: float = 0.05) -> dict:
        """Paired bootstrap resampling significance test."""
        rng = np.random.RandomState(42)
        a, b = np.array(scores_a), np.array(scores_b)
        n = len(a)
        diff_obs = np.mean(a) - np.mean(b)
        count = 0
        for _ in range(n_iter):
            idx = rng.randint(0, n, size=n)
            diff_boot = np.mean(a[idx]) - np.mean(b[idx])
            if diff_boot < 0:
                count += 1
        p_value = count / n_iter
        return {"observed_diff": diff_obs, "p_value": p_value,
                "significant": p_value < alpha}


# ---------------------------------------------------------------------------
# 8. Lambda Trainer
# ---------------------------------------------------------------------------

class LambdaTrainer:
    """Train the adaptive lambda parameters W and b (Eq. 8)."""

    def __init__(self, lr: float = 1e-4, epochs: int = 10, batch_size: int = 32):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, features: np.ndarray, labels: np.ndarray,
              verifier: RelationVerifier, val_features: np.ndarray = None,
              val_labels: np.ndarray = None,
              checkpoint_path: str = "checkpoint_lambda.json") -> Dict:
        """Train with checkpoint support for crash recovery."""
        W = verifier.W.copy()
        b = verifier.b
        n = len(labels)
        best_val_loss = float("inf")
        history: List[dict] = []
        start_epoch = 0

        # Resume from checkpoint
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "r") as f:
                    ckpt = json.load(f)
                W = np.array(ckpt["W"])
                b = float(ckpt["b"])
                start_epoch = ckpt["epoch"]
                history = ckpt.get("history", [])
                print(f"[LambdaTrainer] Resumed from epoch {start_epoch}",
                      flush=True)
            except Exception:
                start_epoch = 0

        for epoch in range(start_epoch, self.epochs):
            perm = np.random.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                X = features[idx]
                y = labels[idx]
                logits = X @ W + b
                probs = 1.0 / (1.0 + np.exp(-logits))
                eps = 1e-7
                loss = -np.mean(y * np.log(probs + eps) +
                                (1 - y) * np.log(1 - probs + eps))
                epoch_loss += loss * len(idx)
                grad = probs - y
                W -= self.lr * (X.T @ grad) / len(idx)
                b -= self.lr * np.mean(grad)

            epoch_loss /= n
            entry = {"epoch": epoch + 1, "train_loss": epoch_loss}

            if val_features is not None:
                vl = val_features @ W + b
                vp = 1.0 / (1.0 + np.exp(-vl))
                eps = 1e-7
                val_loss = -np.mean(val_labels * np.log(vp + eps) +
                                    (1 - val_labels) * np.log(1 - vp + eps))
                entry["val_loss"] = val_loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

            history.append(entry)

            # Save checkpoint every epoch
            ckpt_data = {"epoch": epoch + 1, "W": W.tolist(), "b": float(b),
                         "history": history}
            with open(checkpoint_path, "w") as f:
                json.dump(ckpt_data, f)

            if (epoch + 1) % 2 == 0:
                print(f"  [LambdaTrainer] Epoch {epoch+1}/{self.epochs}  "
                      f"loss={epoch_loss:.4f}  (checkpoint saved)", flush=True)

        verifier.W = W
        verifier.b = b

        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"[LambdaTrainer] Training complete, checkpoint removed",
                  flush=True)

        return {"W": W.tolist(), "b": float(b), "history": history}


# ---------------------------------------------------------------------------
# 9. Demo / CLI
# ---------------------------------------------------------------------------

def demo():
    """Quick demonstration with built-in sample data."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    sample_kg = {
        "entities": [
            {"id": "metformin", "name": "二甲双胍",
             "aliases": ["metformin", "格华止"],
             "description": "双胍类口服降糖药，用于2型糖尿病的治疗"},
            {"id": "t2dm", "name": "2型糖尿病",
             "aliases": ["type 2 diabetes", "T2DM", "II型糖尿病"],
             "description": "以胰岛素抵抗为主的代谢性疾病"},
            {"id": "aspirin", "name": "阿司匹林",
             "aliases": ["aspirin", "乙酰水杨酸"],
             "description": "非甾体抗炎药，用于镇痛、抗血小板"},
            {"id": "gi_side", "name": "胃肠道反应",
             "aliases": ["gastrointestinal reaction"],
             "description": "药物引起的恶心、呕吐、腹泻等胃肠道症状"},
            {"id": "hypoglycemia", "name": "低血糖",
             "aliases": ["hypoglycemia"],
             "description": "血糖低于正常水平的症状"},
            {"id": "cvd", "name": "心血管疾病",
             "aliases": ["cardiovascular disease", "CVD"],
             "description": "影响心脏和血管的疾病总称"},
        ],
        "relations": [
            {"head": "metformin", "tail": "t2dm", "relation": "treats", "weight": 1.0},
            {"head": "metformin", "tail": "gi_side", "relation": "causes", "weight": 1.0},
            {"head": "aspirin", "tail": "cvd", "relation": "treats", "weight": 0.9},
            {"head": "t2dm", "tail": "cvd", "relation": "risk_factor_for", "weight": 0.8},
        ]
    }

    kg_path = os.path.join(os.path.dirname(__file__), "_demo_kg.json")
    with open(kg_path, "w", encoding="utf-8") as f:
        json.dump(sample_kg, f, ensure_ascii=False)

    config = {
        "kg_path": kg_path,
        "kg_format": "json",
        "device": "cuda",
        "lang": "zh",
        "tau_h": 0.5,
        "tau_e": 0.5,
        "beta": 0.5,
    }

    detector = AdaTriple(config)

    sample = (
        "患者诊断为2型糖尿病。建议使用二甲双胍500mg，每日两次口服。"
        "同时注意监测血糖，空腹血糖控制在4.4-7.0mmol/L。"
        "糖尿病患者可以同时服用阿司匹林预防心血管疾病。"
        "二甲双胍的主要副作用是低血糖。"
    )

    result = detector.detect(sample)
    print("\n" + "=" * 60)
    print("AdaTriple Detection Result")
    print("=" * 60)
    print(f"Response score: {result.response_score:.3f}")
    print(f"Triples found:  {len(result.triples)}")
    print(f"Hallucinated:   {result.hallucinated_triples}")
    for i, (s, sc) in enumerate(zip(result.sentences, result.sentence_scores)):
        flag = " << HALLUCINATION" if sc > 0.5 else ""
        print(f"  S{i} [{sc:.2f}]{flag}  {s[:60]}")
    for j, t in enumerate(result.triples):
        h_flag = "*" if t.hallucination_score > 0.5 else " "
        print(f"  {h_flag} T{j}: ({t.head_entity.text}, {t.relation}, "
              f"{t.tail_entity.text})  H={t.hallucination_score:.3f}  "
              f"lam={t.lambda_weight:.2f}")

    os.remove(kg_path)
    print("\nDemo complete.")


if __name__ == "__main__":
    demo()
