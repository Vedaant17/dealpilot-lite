import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";

import EvidenceList from "~/components/EvidenceList.vue";
import WarningList from "~/components/WarningList.vue";
import ExecutionMetadata from "~/components/ExecutionMetadata.vue";
import ScreeningStatus from "~/components/ScreeningStatus.vue";
import ScreeningForm from "~/components/ScreeningForm.vue";
import ScreeningResult from "~/components/ScreeningResult.vue";

describe("EvidenceList", () => {
  it("renders evidence and labels similarity as retrieval similarity", () => {
    const wrapper = mount(EvidenceList, {
      props: {
        items: [
          {
            company_id: "aurora-retail",
            document_id: "doc-1",
            chunk_id: 1,
            chunk_index: 0,
            text_preview: "Evidence text",
            source: "source.txt",
            filename: "source.txt",
            page_start: 1,
            page_end: 1,
            similarity_score: 0.9,
            embedding_model: "text-embedding-3-small",
          },
        ],
      },
    });

    expect(wrapper.text()).toContain("Evidence text");
    expect(wrapper.text()).toContain("Retrieval similarity: 0.9000");
    expect(wrapper.text()).not.toContain("confidence");
  });

  it("renders an empty state", () => {
    const wrapper = mount(EvidenceList, { props: { items: [] } });
    expect(wrapper.text()).toContain("No documentary evidence was returned.");
  });
});

describe("WarningList", () => {
  it("renders only when items exist", () => {
    const empty = mount(WarningList, { props: { title: "Warnings", items: [] } });
    expect(empty.text()).toBe("");

    const populated = mount(WarningList, { props: { title: "Warnings", items: ["one"] } });
    expect(populated.text()).toContain("Warnings");
    expect(populated.text()).toContain("one");
  });
});

describe("ExecutionMetadata", () => {
  it("renders metadata and handles missing tools", () => {
    const wrapper = mount(ExecutionMetadata, {
      props: {
        metadata: {
          duration_ms: 12,
          tool_call_count: 2,
          round_count: 1,
          tools_used: [],
          termination_reason: null,
        },
      },
    });

    expect(wrapper.text()).toContain("12 ms");
    expect(wrapper.text()).toContain("None");
  });
});

describe("ScreeningStatus", () => {
  it("renders loading and completed states", () => {
    const loading = mount(ScreeningStatus, { props: { status: null, loading: true } });
    expect(loading.text()).toContain("Running screening analysis");

    const completed = mount(ScreeningStatus, { props: { status: "completed", loading: false } });
    expect(completed.text()).toContain("Screening completed.");
  });
});

describe("ScreeningForm", () => {
  it("disables submission while loading", () => {
    const wrapper = mount(ScreeningForm, {
      props: {
        company: "aurora-retail",
        query: "question",
        companyOptions: [{ id: "aurora-retail", name: "Aurora Retail Solutions" }],
        loading: true,
        validationError: null,
      },
    });
    expect((wrapper.find("button").element as HTMLButtonElement).disabled).toBe(true);
  });
});

describe("ScreeningResult", () => {
  it("renders narrative, evidence, warnings, limitations, and disclaimers", () => {
    const wrapper = mount(ScreeningResult, {
      props: {
        result: {
          company_id: "aurora-retail",
          company_name: "Aurora Retail Solutions",
          query: "question",
          status: "partial",
          narrative: "Narrative text",
          evidence: [
            {
              company_id: "aurora-retail",
              document_id: "doc-1",
              chunk_id: 1,
              chunk_index: 0,
              text_preview: "Evidence text",
              source: "source.txt",
              filename: "source.txt",
              page_start: 1,
              page_end: 1,
              similarity_score: 0.9,
              embedding_model: "text-embedding-3-small",
            },
          ],
          citation_warnings: ["citation warning"],
          warnings: ["operational warning"],
          limitations: ["limitation"],
          execution: {
            duration_ms: 10,
            tool_call_count: 1,
            round_count: 1,
            tools_used: ["retrieve_company_documents"],
            termination_reason: null,
          },
          disclaimers: ["synthetic data"],
        },
      },
    });

    expect(wrapper.text()).toContain("Narrative text");
    expect(wrapper.text()).toContain("Evidence text");
    expect(wrapper.text()).toContain("citation warning");
    expect(wrapper.text()).toContain("operational warning");
    expect(wrapper.text()).toContain("limitation");
    expect(wrapper.text()).toContain("synthetic data");
  });
});