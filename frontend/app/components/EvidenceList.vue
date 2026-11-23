<script setup lang="ts">
import type { EvidenceItem } from "~/types/screening";

defineProps<{
  items: EvidenceItem[];
}>();
</script>

<template>
  <section v-if="items.length" class="evidence-list">
    <h3>Retrieved evidence</h3>
    <article v-for="item in items" :key="`${item.document_id}-${item.chunk_id}`" class="evidence-card">
      <header class="evidence-card__header">
        <strong>{{ item.filename }}</strong>
        <span>Pages {{ item.page_start }}-{{ item.page_end }}</span>
      </header>
      <p class="evidence-card__meta">
        Document {{ item.document_id }} · Chunk {{ item.chunk_id }} · Source {{ item.source }}
      </p>
      <p class="evidence-card__preview">{{ item.text_preview }}</p>
      <p class="evidence-card__score">
        Retrieval similarity: {{ item.similarity_score.toFixed(4) }} · {{ item.embedding_model }}
      </p>
    </article>
  </section>
  <section v-else class="evidence-list evidence-list--empty">
    <h3>Retrieved evidence</h3>
    <p>No documentary evidence was returned.</p>
  </section>
</template>