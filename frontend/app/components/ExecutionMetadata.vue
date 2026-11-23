<script setup lang="ts">
import type { ExecutionMetadata as ExecutionMetadataType } from "~/types/screening";

defineProps<{
  metadata: ExecutionMetadataType;
}>();
</script>

<template>
  <section class="metadata">
    <h3>Execution metadata</h3>
    <dl>
      <div>
        <dt>Duration</dt>
        <dd>{{ metadata.duration_ms }} ms</dd>
      </div>
      <div>
        <dt>Tool calls</dt>
        <dd>{{ metadata.tool_call_count }}</dd>
      </div>
      <div>
        <dt>Rounds</dt>
        <dd>{{ metadata.round_count }}</dd>
      </div>
      <div>
        <dt>Tools used</dt>
        <dd>{{ metadata.tools_used.join(", ") || "None" }}</dd>
      </div>
      <div v-if="metadata.termination_reason">
        <dt>Termination</dt>
        <dd>{{ metadata.termination_reason }}</dd>
      </div>
    </dl>
  </section>
</template>