<script setup lang="ts">
import { computed } from "vue";

import type { CompanyOption } from "~/utils/companies";

const props = defineProps<{
  company: string;
  query: string;
  companyOptions: CompanyOption[];
  loading: boolean;
  validationError: string | null;
}>();

const emit = defineEmits<{
  "update:company": [value: string];
  "update:query": [value: string];
  submit: [];
}>();

const canSubmit = computed(() => !props.loading && props.company !== "" && props.query.trim().length > 0);
</script>

<template>
  <section class="panel">
    <h2>Screening request</h2>
    <label class="field">
      <span class="field__label">Company</span>
      <select
        class="field__control"
        :value="company"
        :disabled="loading"
        @change="emit('update:company', ($event.target as HTMLSelectElement).value)"
      >
        <option value="" disabled>Select a company</option>
        <option v-for="option in companyOptions" :key="option.id" :value="option.id">
          {{ option.name }}
        </option>
      </select>
    </label>

    <label class="field">
      <span class="field__label">Screening question</span>
      <textarea
        class="field__control field__control--textarea"
        :value="query"
        :disabled="loading"
        maxlength="4000"
        @input="emit('update:query', ($event.target as HTMLTextAreaElement).value)"
      ></textarea>
      <span class="field__hint">{{ query.trim().length }} / 4000 characters</span>
    </label>

    <p v-if="validationError" class="message message--error" role="alert">{{ validationError }}</p>

    <button class="button button--primary" type="button" :disabled="!canSubmit" @click="emit('submit')">
      {{ loading ? "Screening..." : "Run screening" }}
    </button>
  </section>
</template>