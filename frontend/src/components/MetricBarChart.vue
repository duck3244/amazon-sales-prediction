<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart } from 'echarts/charts'
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
  TitleComponent,
} from 'echarts/components'
import type { EChartsOption } from 'echarts'

use([CanvasRenderer, BarChart, GridComponent, TooltipComponent, LegendComponent, TitleComponent])

const props = defineProps<{
  labels: string[]
  series: { name: string; values: number[] }[]
}>()

const option = computed<EChartsOption>(() => ({
  tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
  legend: { data: props.series.map((s) => s.name) },
  grid: { left: 60, right: 20, top: 30, bottom: 30 },
  xAxis: { type: 'category', data: props.labels },
  yAxis: { type: 'value' },
  series: props.series.map((s) => ({
    name: s.name,
    type: 'bar',
    data: s.values,
  })),
}))
</script>

<template>
  <v-chart class="w-full h-72" :option="option" autoresize />
</template>
