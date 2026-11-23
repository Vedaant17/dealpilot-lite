<script setup lang="ts">
import type { ScreeningResponse } from "~/types/screening";

import EvidenceList from "./EvidenceList.vue";
import ExecutionMetadata from "./ExecutionMetadata.vue";
import WarningList from "./WarningList.vue";

defineProps<{
  result: ScreeningResponse;
}>();
</script>

<template>
  <section class="screening-result">
    <header class="screening-result__header">
      <h2>Screening result</h2>
      <p>{{ result.company_name }} · {{ result.query }}</p>
    </header>

    <section class="narrative">
      <h3>Narrative</h3>
      <div class="narrative__body">
        <p v-for="(paragraph, index) in result.narrative.split(/\n{2,}/)" :key="index">
          {{ paragraph }}
        </p>
      </div>
    </section>

    <WarningList title="Citation warnings" :items="result.citation_warnings" />
    <WarningList title="Operational warnings" :items="result.warnings" />
    <WarningList title="Limitations" :items="result.limitations" />

    <EvidenceList :items="result.evidence" />
    <ExecutionMetadata :metadata="result.execution" />

    <section class="disclaimers">
      <h3>Disclaimers</h3>
      <ul>
        <li v-for="(disclaimer, index) in result.disclaimers" :key="index">{{ disclaimer }}</li>
      </ul>
    </section>
  </section>
</template>