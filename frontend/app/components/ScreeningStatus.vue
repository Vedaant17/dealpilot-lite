<script setup lang="ts">
import type { ScreeningStatus } from "~/types/screening";

defineProps<{
  status: ScreeningStatus | null;
  loading: boolean;
}>();
</script>

<template>
  <p v-if="loading" class="status status--loading" role="status">Running screening analysis...</p>
  <p v-else-if="status === 'completed'" class="status status--completed">Screening completed.</p>
  <p v-else-if="status === 'partial'" class="status status--partial">Screening completed with limitations.</p>
  <p v-else-if="status === 'failed'" class="status status--failed">Screening failed.</p>
</template>