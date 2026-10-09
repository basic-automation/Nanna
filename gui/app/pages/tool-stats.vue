<template>
  <div class="h-full flex flex-col">
    <!-- Header -->
    <div class="px-6 py-4 border-b border-white/[0.04]">
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-2xl font-bold text-nanna-text">Tool Stats</h1>
          <p class="text-sm text-nanna-text-muted mt-1">Per-tool performance, latency percentiles, error rates &amp; diagnostics</p>
        </div>
        <div class="flex items-center gap-2">
          <button
            @click="fetchAll"
            class="px-3 py-2 rounded-lg text-sm font-medium glass-chip text-nanna-text-muted hover:text-nanna-text"
          >
            <RefreshCw class="w-4 h-4 inline mr-1" :class="isLoading ? 'animate-spin' : ''" />
            Refresh
          </button>
        </div>
      </div>
    </div>

    <!-- Content -->
    <div class="flex-1 overflow-y-auto p-6 space-y-6">

      <PageState
        v-if="isLoading || !isOnline || loadError || tools.length === 0"
        :state="isLoading ? 'loading' : (!isOnline ? 'offline' : (loadError ? 'error' : 'empty'))"
        :title="isLoading ? 'Loading tool stats…' : (!isOnline ? 'Daemon offline' : (loadError ? 'Could not load tool stats' : 'No tool stats yet'))"
        :description="isLoading
          ? 'Reading per-tool counters from the daemon.'
          : (!isOnline
            ? 'Tool stats live on the daemon. Reconnect after a few turns with tools to populate this dashboard.'
            : (loadError || 'Stats will appear after tools are executed in agent sessions.'))"
        :primary-action="(!isOnline || !!loadError) && !isLoading ? 'Retry' : ''"
        :primary-busy="isLoading"
        @primary="fetchAll"
      />

      <!-- Global Dashboard Cards -->
      <div v-if="tools.length > 0" class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div class="glass-panel rounded-xl p-4">
          <div class="text-xs text-nanna-text-muted mb-1">Total Calls</div>
          <div class="text-2xl font-bold font-mono text-nanna-text">
            {{ global.total_calls?.toLocaleString() ?? 0 }}
          </div>
        </div>
        <div class="glass-panel rounded-xl p-4">
          <div class="text-xs text-nanna-text-muted mb-1">Avg Latency</div>
          <div class="text-2xl font-bold font-mono text-nanna-text">
            {{ formatLatency(global.avg_latency_ms ?? 0) }}
          </div>
        </div>
        <div class="glass-panel rounded-xl p-4">
          <div class="text-xs text-nanna-text-muted mb-1">Success Rate</div>
          <div :class="[
            'text-2xl font-bold font-mono',
            (global.success_rate ?? 1) >= 0.95 ? 'text-nanna-success' :
            (global.success_rate ?? 1) >= 0.8 ? 'text-nanna-warning' :
            'text-nanna-error'
          ]">
            {{ ((global.success_rate ?? 1) * 100).toFixed(1) }}%
          </div>
        </div>
        <div class="glass-panel rounded-xl p-4">
          <div class="text-xs text-nanna-text-muted mb-1">Unique Tools</div>
          <div class="text-2xl font-bold font-mono text-nanna-accent">
            {{ tools.length }}
          </div>
        </div>
      </div>

      <!-- Diagnostics Alerts (Phase 4) -->
      <div v-if="diagnostics.length > 0" class="space-y-2">
        <div
          v-for="(diag, i) in diagnostics" :key="i"
          :class="[
            'flex items-center gap-3 px-4 py-3 rounded-lg border text-sm',
            diag.level === 'error'
              ? 'bg-nanna-error/10 border-nanna-error/30 text-nanna-error'
              : diag.level === 'warning'
                ? 'bg-nanna-warning/10 border-nanna-warning/30 text-nanna-warning'
                : 'bg-nanna-accent/10 border-nanna-accent/30 text-nanna-accent'
          ]"
        >
          <AlertTriangle v-if="diag.level === 'error' || diag.level === 'warning'" class="w-4 h-4 flex-shrink-0" />
          <Info v-else class="w-4 h-4 flex-shrink-0" />
          <span>{{ diag.message }}</span>
        </div>
      </div>

      <!-- Tool vs LLM Time Breakdown -->
      <div v-if="sessionTotals.total_tool_time_ms > 0 || sessionTotals.total_llm_time_ms > 0"
           class="glass-panel rounded-xl p-5">
        <h3 class="text-sm font-semibold text-nanna-text mb-3">Time Breakdown: Tool vs LLM</h3>
        <div class="flex items-center gap-4">
          <div class="flex-1">
            <div class="flex justify-between text-xs text-nanna-text-muted mb-1">
              <span>Tool Time</span>
              <span>{{ formatDuration(sessionTotals.total_tool_time_ms) }}</span>
            </div>
            <div class="h-4 glass-well rounded-full overflow-hidden flex">
              <div
                class="bg-nanna-accent h-full transition-all duration-500"
                :style="{ width: toolTimePct + '%' }"
              />
              <div
                class="bg-nanna-primary h-full transition-all duration-500"
                :style="{ width: llmTimePct + '%' }"
              />
            </div>
            <div class="flex justify-between text-xs text-nanna-text-muted mt-1">
              <span>LLM Time</span>
              <span>{{ formatDuration(sessionTotals.total_llm_time_ms) }}</span>
            </div>
          </div>
          <div class="text-right">
            <div class="flex gap-3 text-[10px] text-nanna-text-dim">
              <span class="flex items-center gap-1">
                <span class="w-2 h-2 rounded-full bg-nanna-accent inline-block" />
                Tool: {{ toolTimePct.toFixed(0) }}%
              </span>
              <span class="flex items-center gap-1">
                <span class="w-2 h-2 rounded-full bg-nanna-primary inline-block" />
                LLM: {{ llmTimePct.toFixed(0) }}%
              </span>
            </div>
          </div>
        </div>
      </div>

      <!-- Hourly Activity Graph -->
      <div v-if="hourlyBuckets.length > 0" class="glass-panel rounded-xl p-5">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-sm font-semibold text-nanna-text">Tool Calls — Last 24 Hours</h3>
          <div class="flex gap-2 text-[10px] text-nanna-text-dim">
            <span class="flex items-center gap-1"><span class="w-2 h-2 rounded-full bg-nanna-success inline-block" /> Success</span>
            <span class="flex items-center gap-1"><span class="w-2 h-2 rounded-full bg-nanna-error inline-block" /> Failed</span>
          </div>
        </div>
        <div class="flex items-end gap-[2px] h-32">
          <div
            v-for="(b, i) in hourlyBuckets" :key="i"
            class="flex-1 flex flex-col justify-end relative group"
          >
            <!-- Tooltip -->
            <div class="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block z-10">
              <div class="glass-strong rounded-lg px-2 py-1 text-[10px] text-nanna-text whitespace-nowrap">
                <div class="font-mono">{{ b.period?.slice(11, 16) || '?' }}</div>
                <div>{{ b.call_count }} calls · {{ formatLatency(b.avg_duration_ms) }} avg</div>
              </div>
            </div>
            <!-- Success bar -->
            <div
              class="bg-nanna-success/70 rounded-t-sm transition-all duration-300"
              :style="{ height: barHeight(b.success_count, hourlyMax) + 'px' }"
            />
            <!-- Failure bar -->
            <div
              v-if="b.failure_count > 0"
              class="bg-nanna-error/70 rounded-t-sm transition-all duration-300"
              :style="{ height: barHeight(b.failure_count, hourlyMax) + 'px' }"
            />
          </div>
        </div>
        <!-- X-axis labels -->
        <div class="flex justify-between mt-1 text-[9px] text-nanna-text-dim font-mono">
          <span>{{ hourlyBuckets[0]?.period?.slice(11, 16) || '' }}</span>
          <span>{{ hourlyBuckets[Math.floor(hourlyBuckets.length / 2)]?.period?.slice(11, 16) || '' }}</span>
          <span>{{ hourlyBuckets[hourlyBuckets.length - 1]?.period?.slice(11, 16) || '' }}</span>
        </div>
      </div>

      <!-- Daily Trend Graph -->
      <div v-if="dailyBuckets.length > 0" class="glass-panel rounded-xl p-5">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-sm font-semibold text-nanna-text">Daily Trend — Last 30 Days</h3>
          <div class="flex gap-2 text-[10px] text-nanna-text-dim">
            <span class="flex items-center gap-1"><span class="w-2 h-2 rounded-full bg-nanna-accent inline-block" /> Avg Latency</span>
            <span class="flex items-center gap-1"><span class="w-2 h-2 rounded-full bg-nanna-primary inline-block" /> Calls</span>
          </div>
        </div>
        <!-- Dual-axis: bars for calls, line for latency -->
        <div class="flex items-end gap-[2px] h-32">
          <div
            v-for="(b, i) in dailyBuckets" :key="i"
            class="flex-1 flex flex-col justify-end relative group"
          >
            <!-- Tooltip -->
            <div class="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block z-10">
              <div class="glass-strong rounded-lg px-2 py-1 text-[10px] text-nanna-text whitespace-nowrap">
                <div class="font-mono">{{ b.period }}</div>
                <div>{{ b.call_count }} calls · {{ formatLatency(b.avg_duration_ms) }} avg</div>
                <!-- The bucket's column keeps the MAX of its calls, not a percentile. -->
                <div>Slowest: {{ formatLatency(b.p95_duration_ms) }}</div>
              </div>
            </div>
            <!-- Call count bar -->
            <div
              class="bg-nanna-primary/50 rounded-t-sm transition-all duration-300"
              :style="{ height: barHeight(b.call_count, dailyMax) + 'px' }"
            />
            <!-- Latency dot overlay -->
            <div
              class="absolute bottom-0 left-1/2 -translate-x-1/2 w-1.5 h-1.5 rounded-full bg-nanna-accent"
              :style="{ bottom: barHeight(b.avg_duration_ms, dailyLatencyMax) + 'px' }"
            />
          </div>
        </div>
        <!-- X-axis labels -->
        <div class="flex justify-between mt-1 text-[9px] text-nanna-text-dim font-mono">
          <span>{{ dailyBuckets[0]?.period?.slice(5) || '' }}</span>
          <span>{{ dailyBuckets[Math.floor(dailyBuckets.length / 2)]?.period?.slice(5) || '' }}</span>
          <span>{{ dailyBuckets[dailyBuckets.length - 1]?.period?.slice(5) || '' }}</span>
        </div>
      </div>

      <!-- Sortable Tool Performance Table -->
      <div v-if="tools.length > 0" class="glass-panel rounded-xl overflow-hidden">
        <div class="px-5 py-3 border-b border-white/[0.04] flex items-center justify-between">
          <h3 class="text-sm font-semibold text-nanna-text">Tool Performance</h3>
          <input
            v-model="searchQuery"
            type="text"
            placeholder="Filter tools..."
            class="px-3 py-1.5 rounded-lg text-sm glass-well text-nanna-text placeholder-nanna-text-dim focus:outline-none focus:border-nanna-accent/40 w-48"
          />
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="border-b border-white/[0.04] text-nanna-text-muted text-xs uppercase tracking-wider">
                <th class="px-5 py-3 text-left cursor-pointer hover:text-nanna-text" @click="sortBy('name')">
                  Tool {{ sortIcon('name') }}
                </th>
                <th class="px-5 py-3 text-right cursor-pointer hover:text-nanna-text" @click="sortBy('call_count')">
                  Calls {{ sortIcon('call_count') }}
                </th>
                <th class="px-5 py-3 text-right cursor-pointer hover:text-nanna-text" @click="sortBy('p50_latency_ms')">
                  P50 {{ sortIcon('p50_latency_ms') }}
                </th>
                <th class="px-5 py-3 text-right cursor-pointer hover:text-nanna-text" @click="sortBy('p95_latency_ms')">
                  P95 {{ sortIcon('p95_latency_ms') }}
                </th>
                <th class="px-5 py-3 text-right cursor-pointer hover:text-nanna-text" @click="sortBy('success_rate')">
                  Success {{ sortIcon('success_rate') }}
                </th>
                <th class="px-5 py-3 text-right cursor-pointer hover:text-nanna-text" @click="sortBy('avg_output_size')">
                  Avg Output {{ sortIcon('avg_output_size') }}
                </th>
                <th class="px-5 py-3 text-center">Status</th>
              </tr>
            </thead>
            <tbody>
              <template v-for="tool in filteredSortedTools" :key="tool.name">
                <tr
                  class="border-b border-white/[0.03] hover:bg-white/[0.02] transition-colors cursor-pointer"
                  @click="toggleToolDetail(tool.name)"
                >
                  <td class="px-5 py-3 text-nanna-text font-mono font-medium">
                    <div class="flex items-center gap-2">
                      <svg
                        class="w-3 h-3 text-nanna-text-dim transition-transform"
                        :class="{ 'rotate-90': expandedTool === tool.name }"
                        viewBox="0 0 10 10" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"
                      >
                        <path d="M3 1.5l4 3.5-4 3.5" />
                      </svg>
                      {{ tool.name }}
                    </div>
                  </td>
                  <td class="px-5 py-3 text-right text-nanna-text font-mono">
                    {{ tool.call_count.toLocaleString() }}
                  </td>
                  <td class="px-5 py-3 text-right text-nanna-text font-mono">
                    {{ formatLatency(tool.p50_latency_ms) }}
                  </td>
                  <td :class="[
                    'px-5 py-3 text-right font-mono',
                    tool.p95_latency_ms > 5000 ? 'text-nanna-error' : 'text-nanna-text'
                  ]">
                    {{ formatLatency(tool.p95_latency_ms) }}
                  </td>
                  <td :class="[
                    'px-5 py-3 text-right font-mono',
                    tool.success_rate >= 0.95 ? 'text-nanna-success' :
                    tool.success_rate >= 0.8 ? 'text-nanna-warning' :
                    'text-nanna-error'
                  ]">
                    {{ (tool.success_rate * 100).toFixed(1) }}%
                  </td>
                  <td class="px-5 py-3 text-right text-nanna-text-muted font-mono">
                    {{ formatSize(tool.avg_output_size) }}
                  </td>
                  <td class="px-5 py-3 text-center">
                    <div class="flex items-center justify-center gap-1">
                      <span
                        v-if="tool.p95_latency_ms > 5000"
                        class="text-[10px] px-1.5 py-0.5 rounded bg-nanna-error/20 text-nanna-error"
                      >🐌 SLOW</span>
                      <span
                        v-if="tool.success_rate < 0.8"
                        class="text-[10px] px-1.5 py-0.5 rounded bg-nanna-error/20 text-nanna-error"
                      >⚠️ ERRORS</span>
                      <span
                        v-if="tool.p95_latency_ms <= 5000 && tool.success_rate >= 0.8"
                        class="text-[10px] px-1.5 py-0.5 rounded bg-nanna-success/20 text-nanna-success"
                      >OK</span>
                    </div>
                  </td>
                </tr>
                <!-- Expanded error detail row -->
                <tr v-if="expandedTool === tool.name" class="glass-well">
                  <td colspan="7" class="px-5 py-4">
                    <div class="space-y-4">
                      <!-- Error Summary -->
                      <div class="flex items-center gap-6 text-xs">
                        <div>
                          <span class="text-nanna-text-dim">Total Calls:</span>
                          <span class="ml-1 text-nanna-text font-mono">{{ tool.call_count }}</span>
                        </div>
                        <div>
                          <span class="text-nanna-text-dim">Successes:</span>
                          <span class="ml-1 text-nanna-success font-mono">{{ tool.success_count }}</span>
                        </div>
                        <div>
                          <span class="text-nanna-text-dim">Failures:</span>
                          <span class="ml-1 text-nanna-error font-mono">{{ tool.failure_count }}</span>
                        </div>
                        <div>
                          <span class="text-nanna-text-dim">Error Rate:</span>
                          <span class="ml-1 font-mono" :class="tool.success_rate < 0.8 ? 'text-nanna-error' : 'text-nanna-warning'">
                            {{ ((1 - tool.success_rate) * 100).toFixed(1) }}%
                          </span>
                        </div>
                      </div>

                      <!-- Top Errors -->
                      <div v-if="tool.top_errors?.length > 0">
                        <h4 class="text-xs text-nanna-text-muted mb-2 uppercase tracking-wider">Top Errors</h4>
                        <div class="space-y-1.5">
                          <div
                            v-for="([errorMsg, count], idx) in tool.top_errors"
                            :key="idx"
                            class="flex items-start gap-3 text-xs glass-well rounded-lg px-3 py-2"
                          >
                            <span class="text-nanna-error font-mono font-bold shrink-0">×{{ count }}</span>
                            <div class="flex-1 min-w-0">
                              <pre class="text-nanna-text-muted font-mono whitespace-pre-wrap break-all text-[11px]">{{ errorMsg }}</pre>
                            </div>
                            <!-- Error percentage bar -->
                            <div class="shrink-0 w-16 flex items-center gap-1">
                              <div class="flex-1 h-1.5 glass-well rounded-full overflow-hidden">
                                <div
                                  class="h-full bg-nanna-error/60 rounded-full"
                                  :style="{ width: `${(count / tool.failure_count * 100).toFixed(0)}%` }"
                                />
                              </div>
                              <span class="text-[9px] text-nanna-text-dim font-mono">
                                {{ (count / tool.failure_count * 100).toFixed(0) }}%
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                      <div v-else-if="tool.failure_count > 0" class="text-xs text-nanna-text-dim italic">
                        {{ tool.failure_count }} error(s) recorded but no error messages captured
                      </div>

                      <!-- Recent Failures from log -->
                      <div v-if="toolCallLog.length > 0 && expandedTool === tool.name">
                        <h4 class="text-xs text-nanna-text-muted mb-2 uppercase tracking-wider">Recent Failures</h4>
                        <div class="space-y-1">
                          <div
                            v-for="(entry, idx) in toolCallLog.filter(e => !e.success).slice(0, 10)"
                            :key="idx"
                            class="flex items-start gap-3 text-[11px] glass-well rounded px-3 py-1.5"
                          >
                            <span class="text-nanna-text-dim font-mono shrink-0">{{ entry.created_at?.slice(5, 16) || '—' }}</span>
                            <span class="text-nanna-text-dim font-mono shrink-0">{{ formatLatency(entry.duration_ms) }}</span>
                            <pre class="flex-1 text-nanna-error/80 font-mono whitespace-pre-wrap break-all">{{ entry.error_message || '(no error message)' }}</pre>
                          </div>
                        </div>
                      </div>

                      <!-- Latency Distribution -->
                      <div>
                        <h4 class="text-xs text-nanna-text-muted mb-2 uppercase tracking-wider">Latency Percentiles</h4>
                        <div class="flex items-end gap-1 h-8">
                          <div v-for="(val, label) in { P50: tool.p50_latency_ms, P95: tool.p95_latency_ms, P99: tool.p99_latency_ms }" :key="label" class="flex flex-col items-center gap-0.5">
                            <div
                              class="w-8 rounded-t"
                              :class="val > 5000 ? 'bg-nanna-error/40' : val > 1000 ? 'bg-nanna-warning/40' : 'bg-nanna-accent/40'"
                              :style="{ height: `${Math.max(4, Math.min(32, val / Math.max(tool.p99_latency_ms, 1) * 32))}px` }"
                            />
                            <span class="text-[9px] text-nanna-text-dim font-mono">{{ label }}</span>
                            <span class="text-[9px] text-nanna-text font-mono">{{ formatLatency(val) }}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </td>
                </tr>
              </template>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Session Totals -->
      <div v-if="sessionTotals.total_iterations > 0" class="glass-panel rounded-xl p-5">
        <h3 class="text-sm font-semibold text-nanna-text mb-3">Session Aggregates</h3>
        <div class="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div class="glass-well rounded-lg p-3">
            <div class="text-xs text-nanna-text-muted mb-1">Total Iterations</div>
            <div class="text-lg font-bold font-mono text-nanna-text">{{ sessionTotals.total_iterations.toLocaleString() }}</div>
          </div>
          <div class="glass-well rounded-lg p-3">
            <div class="text-xs text-nanna-text-muted mb-1">Total Tool Calls</div>
            <div class="text-lg font-bold font-mono text-nanna-text">{{ sessionTotals.total_tool_calls.toLocaleString() }}</div>
          </div>
          <div class="glass-well rounded-lg p-3">
            <div class="text-xs text-nanna-text-muted mb-1">Input Tokens</div>
            <div class="text-lg font-bold font-mono text-nanna-accent">{{ formatTokens(sessionTotals.total_input_tokens) }}</div>
          </div>
          <div class="glass-well rounded-lg p-3">
            <div class="text-xs text-nanna-text-muted mb-1">Output Tokens</div>
            <div class="text-lg font-bold font-mono text-nanna-primary">{{ formatTokens(sessionTotals.total_output_tokens) }}</div>
          </div>
          <div class="glass-well rounded-lg p-3">
            <div class="text-xs text-nanna-text-muted mb-1">Tool Time</div>
            <div class="text-lg font-bold font-mono text-nanna-text">{{ formatDuration(sessionTotals.total_tool_time_ms) }}</div>
          </div>
          <div class="glass-well rounded-lg p-3">
            <div class="text-xs text-nanna-text-muted mb-1">LLM Time</div>
            <div class="text-lg font-bold font-mono text-nanna-text">{{ formatDuration(sessionTotals.total_llm_time_ms) }}</div>
          </div>
        </div>
      </div>

    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { invoke } from '@tauri-apps/api/core'
import { RefreshCw, Wrench, AlertTriangle, Info } from '@lucide/vue'

const { isOnline } = useBackend()
const toast = useToast()
const { addNotification } = useNotificationCenter()

interface ToolStat {
  name: string
  call_count: number
  success_count: number
  failure_count: number
  success_rate: number
  avg_latency_ms: number
  p50_latency_ms: number
  p95_latency_ms: number
  p99_latency_ms: number
  avg_output_size: number
  last_called: number | null
  top_errors: [string, number][]
}

interface SessionTotals {
  total_iterations: number
  total_tool_calls: number
  total_tool_time_ms: number
  total_llm_time_ms: number
  total_input_tokens: number
  total_output_tokens: number
}

interface GlobalStats {
  total_calls: number
  avg_latency_ms: number
  success_rate: number
  slowest_tools: ToolStat[]
  most_used_tools: ToolStat[]
  most_failed_tools: ToolStat[]
  session_totals: SessionTotals
}

interface Diagnostic {
  level: 'error' | 'warning' | 'info'
  message: string
}

interface TimeBucket {
  tool_name: string
  period: string
  call_count: number
  success_count: number
  failure_count: number
  total_duration_ms: number
  avg_duration_ms: number
  p95_duration_ms: number
}

const tools = ref<ToolStat[]>([])
const global = ref<GlobalStats>({
  total_calls: 0,
  avg_latency_ms: 0,
  success_rate: 1,
  slowest_tools: [],
  most_used_tools: [],
  most_failed_tools: [],
  session_totals: {
    total_iterations: 0,
    total_tool_calls: 0,
    total_tool_time_ms: 0,
    total_llm_time_ms: 0,
    total_input_tokens: 0,
    total_output_tokens: 0,
  },
})
const hourlyBuckets = ref<TimeBucket[]>([])
const dailyBuckets = ref<TimeBucket[]>([])
const isLoading = ref(false)
// Referenced by the PageState block in the template and assigned in loadStats();
// it was never declared, so rendering this page threw "loadError is not defined"
// and the ErrorBoundary swallowed the whole panel.
const loadError = ref<string | null>(null)
const sortField = ref<string>('call_count')
const sortAsc = ref(false)
const searchQuery = ref('')
const expandedTool = ref<string | null>(null)
const toolCallLog = ref<any[]>([])
let refreshInterval: ReturnType<typeof setInterval> | null = null

async function toggleToolDetail(name: string) {
  if (expandedTool.value === name) {
    expandedTool.value = null
    toolCallLog.value = []
    return
  }
  expandedTool.value = name
  toolCallLog.value = []
  try {
    const result = await invoke<{ entries: any[] }>('get_tool_call_log', { toolName: name, limit: 50 })
    toolCallLog.value = result.entries || []
  } catch (e) {
    console.error('Failed to load tool call log:', e)
  }
}

const sessionTotals = computed(() => global.value.session_totals ?? {
  total_iterations: 0,
  total_tool_calls: 0,
  total_tool_time_ms: 0,
  total_llm_time_ms: 0,
  total_input_tokens: 0,
  total_output_tokens: 0,
})

const totalTime = computed(() => sessionTotals.value.total_tool_time_ms + sessionTotals.value.total_llm_time_ms)
const toolTimePct = computed(() => totalTime.value > 0 ? (sessionTotals.value.total_tool_time_ms / totalTime.value) * 100 : 50)
const llmTimePct = computed(() => totalTime.value > 0 ? (sessionTotals.value.total_llm_time_ms / totalTime.value) * 100 : 50)

// Graph helpers
const hourlyMax = computed(() => Math.max(1, ...hourlyBuckets.value.map(b => b.call_count)))
const dailyMax = computed(() => Math.max(1, ...dailyBuckets.value.map(b => b.call_count)))
const dailyLatencyMax = computed(() => Math.max(1, ...dailyBuckets.value.map(b => b.avg_duration_ms)))

function barHeight(value: number, max: number): number {
  if (max <= 0) return 0
  return Math.max(1, Math.round((value / max) * 120)) // max 120px (leaving room in the 128px container)
}

// Phase 4: Diagnostics
const diagnostics = computed<Diagnostic[]>(() => {
  const diags: Diagnostic[] = []

  // Tools with P95 > 5s
  const slowTools = tools.value.filter(t => t.p95_latency_ms > 5000)
  if (slowTools.length > 0) {
    const names = slowTools.map(t => t.name).join(', ')
    diags.push({
      level: 'warning',
      message: `Slow tools (P95 > 5s): ${names}`,
    })
  }

  // Tools with error rate > 20%
  const failingTools = tools.value.filter(t => t.call_count >= 2 && t.success_rate < 0.8)
  if (failingTools.length > 0) {
    const names = failingTools.map(t => `${t.name} (${(t.success_rate * 100).toFixed(0)}%)`).join(', ')
    diags.push({
      level: 'error',
      message: `High error rate tools (>20% failures): ${names}`,
    })
  }

  // Tool-vs-LLM imbalance: tool time > 80% of total
  if (totalTime.value > 10_000 && toolTimePct.value > 80) {
    diags.push({
      level: 'info',
      message: `Tool execution dominates (${toolTimePct.value.toFixed(0)}% of total time). Consider tool optimizations or caching.`,
    })
  }

  // LLM-vs-Tool imbalance: LLM time > 90% of total
  if (totalTime.value > 10_000 && llmTimePct.value > 90) {
    diags.push({
      level: 'info',
      message: `LLM time dominates (${llmTimePct.value.toFixed(0)}% of total time). Tools are efficient, but consider cheaper models or caching.`,
    })
  }

  return diags
})

const filteredSortedTools = computed(() => {
  let filtered = tools.value
  if (searchQuery.value) {
    const q = searchQuery.value.toLowerCase()
    filtered = filtered.filter(t => t.name.toLowerCase().includes(q))
  }
  return [...filtered].sort((a, b) => {
    const field = sortField.value as keyof ToolStat
    const aVal = a[field] ?? 0
    const bVal = b[field] ?? 0
    if (typeof aVal === 'string' && typeof bVal === 'string') {
      return sortAsc.value ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal)
    }
    return sortAsc.value ? Number(aVal) - Number(bVal) : Number(bVal) - Number(aVal)
  })
})

function sortBy(field: string) {
  if (sortField.value === field) {
    sortAsc.value = !sortAsc.value
  } else {
    sortField.value = field
    sortAsc.value = field === 'name' // name defaults ascending
  }
}

function sortIcon(field: string): string {
  if (sortField.value !== field) return ''
  return sortAsc.value ? '↑' : '↓'
}

async function fetchAll() {
  isLoading.value = true
  loadError.value = null
  const errors: string[] = []

  // Fetch each endpoint independently so one failure doesn't block the rest
  try {
    const result = await invoke<{ tools: ToolStat[]; error?: string }>('get_tool_stats')
    if (result.error) errors.push(`tool_stats: ${result.error}`)
    tools.value = result.tools || []
  } catch (e) {
    errors.push(`tool_stats: ${e}`)
  }

  try {
    const result = await invoke<GlobalStats & { error?: string }>('get_global_stats')
    if (result.error) errors.push(`global_stats: ${result.error}`)
    global.value = result
  } catch (e) {
    errors.push(`global_stats: ${e}`)
  }

  try {
    const result = await invoke<{ buckets: TimeBucket[]; error?: string }>('get_tool_stats_hourly', { hours: 24 })
    if (result.error) errors.push(`hourly_stats: ${result.error}`)
    hourlyBuckets.value = result.buckets || []
  } catch (e) {
    errors.push(`hourly_stats: ${e}`)
  }

  try {
    const result = await invoke<{ buckets: TimeBucket[]; error?: string }>('get_tool_stats_daily', { days: 30 })
    if (result.error) errors.push(`daily_stats: ${result.error}`)
    dailyBuckets.value = result.buckets || []
  } catch (e) {
    errors.push(`daily_stats: ${e}`)
  }

  if (errors.length > 0) {
    console.error('Tool stats fetch errors:', errors)
    loadError.value = `${errors.length} endpoint${errors.length > 1 ? 's' : ''} failed`
    addNotification({
      type: 'error',
      title: 'Tool Stats Error',
      summary: loadError.value,
      detail: errors.join('\n'),
      source: 'tool-stats',
    })
  }

  isLoading.value = false
}

function formatLatency(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`
  return `${(ms / 60_000).toFixed(1)}m`
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}K`
  return `${(bytes / (1024 * 1024)).toFixed(1)}M`
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return n.toString()
}

onMounted(async () => {
  await fetchAll()
  refreshInterval = setInterval(fetchAll, 30_000)
})

onUnmounted(() => {
  if (refreshInterval) clearInterval(refreshInterval)
})
</script>
