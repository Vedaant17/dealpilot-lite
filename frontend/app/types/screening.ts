export type ScreeningStatus = "completed" | "partial" | "failed";

export interface ScreeningRequest {
  company_id: string;
  query: string;
}

export interface EvidenceItem {
  company_id: string;
  document_id: string;
  chunk_id: number;
  chunk_index: number;
  text_preview: string;
  source: string;
  filename: string;
  page_start: number;
  page_end: number;
  similarity_score: number;
  embedding_model: string;
}

export interface ExecutionMetadata {
  duration_ms: number;
  tool_call_count: number;
  round_count: number;
  tools_used: string[];
  termination_reason: string | null;
}

export interface ScreeningResponse {
  company_id: string;
  company_name: string;
  query: string;
  status: ScreeningStatus;
  narrative: string;
  evidence: EvidenceItem[];
  citation_warnings: string[];
  warnings: string[];
  limitations: string[];
  execution: ExecutionMetadata;
  disclaimers: string[];
}

export interface ScreeningError {
  status: number | null;
  message: string;
}

export interface ScreeningResult {
  ok: boolean;
  data?: ScreeningResponse;
  error?: ScreeningError;
}