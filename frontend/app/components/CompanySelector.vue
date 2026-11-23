<script setup lang="ts">
import type { CompanyOption } from "~/utils/companies";

defineProps<{
  modelValue: string;
  options: CompanyOption[];
  disabled?: boolean;
}>();

const emit = defineEmits<{
  "update:modelValue": [value: string];
}>();
</script>

<template>
  <label class="field">
    <span class="field__label">Company</span>
    <select
      class="field__control"
      :value="modelValue"
      :disabled="disabled"
      @change="emit('update:modelValue', ($event.target as HTMLSelectElement).value)"
    >
      <option value="" disabled>Select a company</option>
      <option v-for="option in options" :key="option.id" :value="option.id">
        {{ option.name }}
      </option>
    </select>
  </label>
</template>