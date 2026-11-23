<script setup lang="ts">
import { computed, ref } from "vue";

import { COMPANY_OPTIONS } from "~/utils/companies";
import { useScreeningApi } from "~/composables/useScreeningApi";
import type { ScreeningResponse } from "~/types/screening";

const company = ref("");
const query = ref("");
const loading = ref(false);
const validationError = ref<string | null>(null);
const errorMessage = ref<string | null>(null);
const result = ref<ScreeningResponse | null>(null);
const latestFailed = ref(false);

const api = useScreeningApi();
let activeRequestId = 0;

const canSubmit = computed(
  () => !loading.value && company.value !== "" && query.value.trim().length > 0 && query.value.trim().length <= 4000,
);

const status = computed(() => (loading.value ? "loading" : result.value?.status ?? "idle"));

async function submitScreening() {
  if (!canSubmit.value) {
    validationError.value =
      company.value === ""
        ? "Select a company."
        : query.value.trim().length === 0
          ? "Enter a screening question."
          : "Query must be 4000 characters or fewer.";
    return;
  }

  validationError.value = null;
  errorMessage.value = null;
  latestFailed.value = false;
  loading.value = true;
  const requestId = ++activeRequestId;

  const response = await api.submitScreening({
    company_id: company.value,
    query: query.value.trim(),
  });

  if (requestId !== activeRequestId) {
    return;
  }

  loading.value = false;

  if (response.ok && response.data) {
    result.value = response.data;
  } else {
    latestFailed.value = true;
    errorMessage.value = response.error?.message ?? "Screening request failed.";
  }
}
</script>

<template>
  <main class="workspace">
    <header class="workspace__header">
      <div>
        <p class="eyebrow">DealPilot Lite</p>
        <h1>Private Equity Screening Workspace</h1>
        <p class="workspace__intro">
          Run an AI-assisted screening analysis using deterministic financial tools and retrieved company documents.
        </p>
      </div>
      <p class="notice">
        Synthetic fictional company data. Screening output is decision support, not investment advice.
      </p>
    </header>

    <section class="workspace__grid">
      <section class="panel panel--form">
        <h2>Screening request</h2>
        <label class="field">
          <span class="field__label">Company</span>
          <select v-model="company" class="field__control" :disabled="loading">
            <option value="" disabled>Select a company</option>
            <option v-for="option in COMPANY_OPTIONS" :key="option.id" :value="option.id">
              {{ option.name }}
            </option>
          </select>
        </label>

        <label class="field">
          <span class="field__label">Screening question</span>
          <textarea
            v-model="query"
            class="field__control field__control--textarea"
            maxlength="4000"
            :disabled="loading"
            placeholder="Ask about financial performance, valuation, risks, or supporting documents."
          ></textarea>
          <span class="field__hint">{{ query.trim().length }} / 4000 characters</span>
        </label>

        <p v-if="validationError" class="message message--error" role="alert">{{ validationError }}</p>
        <p v-if="errorMessage" class="message message--error" role="alert">{{ errorMessage }}</p>
        <p v-if="latestFailed && result" class="message message--warning" role="status">
          The latest screening failed. The previous result remains below.
        </p>

        <button class="button button--primary" type="button" :disabled="!canSubmit" @click="submitScreening">
          {{ loading ? "Screening..." : "Run screening" }}
        </button>
      </section>

      <section class="panel panel--status">
        <h2>Status</h2>
        <p v-if="loading" class="status status--loading" role="status">Running screening analysis...</p>
        <p v-else-if="status === 'completed'" class="status status--completed">Screening completed.</p>
        <p v-else-if="status === 'partial'" class="status status--partial">Screening completed with limitations.</p>
        <p v-else-if="status === 'failed'" class="status status--failed">Screening failed.</p>
        <p v-else class="status">Select a company and submit a screening question.</p>
      </section>
    </section>

    <ScreeningResult v-if="result" :result="result" />
  </main>
</template>