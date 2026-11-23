import { ofetch } from "ofetch";

import type { ScreeningError, ScreeningRequest, ScreeningResponse, ScreeningResult } from "~/types/screening";

const DEFAULT_API_BASE = "http://127.0.0.1:8000";
const REQUEST_TIMEOUT_MS = 120000;

function resolveApiBase(apiBase?: string): string {
  if (apiBase) {
    return apiBase.replace(/\/+$/, "");
  }

  try {
    const runtimeConfig = useRuntimeConfig();
    const configured = runtimeConfig.public?.apiBase;
    if (typeof configured === "string" && configured.trim()) {
      return configured.replace(/\/+$/, "");
    }
  } catch {
    // Unit tests may call this composable without a Nuxt runtime context.
  }

  return DEFAULT_API_BASE;
}

function toScreeningError(error: unknown): ScreeningError {
  const candidate = error as {
    response?: { status?: number; _data?: { detail?: unknown } };
    statusCode?: number;
    status?: number;
    data?: { detail?: unknown };
    message?: string;
  };

  const status = candidate.response?.status ?? candidate.statusCode ?? candidate.status ?? null;
  const detail = candidate.response?._data?.detail ?? candidate.data?.detail;

  if (status === 404) {
    return { status, message: "The selected company was not found." };
  }
  if (status === 422) {
    return { status, message: "The screening request is invalid." };
  }
  if (status === 502) {
    return { status, message: "The model service is unavailable. Please try again later." };
  }
  if (status === 503) {
    return { status, message: "Backend screening infrastructure is unavailable." };
  }
  if (status === 504) {
    return { status, message: "The screening request timed out." };
  }
  if (status !== null) {
    return { status, message: "The screening request could not be completed." };
  }
  if (detail) {
    return { status, message: "The screening request could not be completed." };
  }
  return { status: null, message: "Unable to reach the screening service." };
}

export function useScreeningApi(apiBase?: string) {
  const resolvedBase = resolveApiBase(apiBase);

  async function submitScreening(request: ScreeningRequest): Promise<ScreeningResult> {
    try {
      const data = await ofetch<ScreeningResponse>("/api/v1/screenings", {
        baseURL: resolvedBase,
        method: "POST",
        body: request,
        timeout: REQUEST_TIMEOUT_MS,
      });
      return { ok: true, data };
    } catch (error) {
      return { ok: false, error: toScreeningError(error) };
    }
  }

  return {
    submitScreening,
  };
}