import { describe, expect, it, vi, beforeEach } from "vitest";

vi.mock("ofetch", () => ({
  ofetch: vi.fn(),
}));

import { ofetch } from "ofetch";
import { useScreeningApi } from "~/composables/useScreeningApi";

const ofetchMock = vi.mocked(ofetch);
import type { ScreeningRequest } from "~/types/screening";

const request: ScreeningRequest = {
  company_id: "aurora-retail",
  query: "Screen Aurora",
};

describe("useScreeningApi", () => {
  beforeEach(() => {
    ofetchMock.mockReset();
  });

  it("submits the expected request", async () => {
    ofetchMock.mockResolvedValue({ status: "completed" });
    const api = useScreeningApi("http://example.test");
    const result = await api.submitScreening(request);

    expect(ofetchMock).toHaveBeenCalledWith("/api/v1/screenings", {
      baseURL: "http://example.test",
      method: "POST",
      body: request,
      timeout: 120000,
    });
    expect(result.ok).toBe(true);
  });

  it("returns structured HTTP errors", async () => {
    ofetchMock.mockRejectedValue({
      response: { status: 404, _data: { detail: "Company not found" } },
    });
    const api = useScreeningApi("http://example.test");
    const result = await api.submitScreening(request);

    expect(result.ok).toBe(false);
    expect(result.error).toEqual({ status: 404, message: "The selected company was not found." });
  });

  it("returns a network error when no response is available", async () => {
    ofetchMock.mockRejectedValue(new Error("fetch failed"));
    const api = useScreeningApi("http://example.test");
    const result = await api.submitScreening(request);

    expect(result.ok).toBe(false);
    expect(result.error).toEqual({ status: null, message: "Unable to reach the screening service." });
  });
});