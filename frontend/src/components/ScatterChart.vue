<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { ScatterChart, LineChart } from 'echarts/charts'
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
  TitleComponent,
} from 'echarts/components'
import type { EChartsOption } from 'echarts'

use([CanvasRenderer, ScatterChart, LineChart, GridComponent, TooltipComponent, LegendComponent, TitleComponent])

const props = defineProps<{ actuals: number[]; predictions: number[] }>()

const option = computed<EChartsOption>(() => {
  const points = props.actuals.map((a, i) => [a, props.predictions[i] ?? 0])
  const all = points.flat()
  const lo = Math.min(...all)
  const hi = Math.max(...all)
  return {
    tooltip: { trigger: 'item', formatter: ({ value }: any) =>
      `actual: ${(value as number[])[0].toFixed(2)}<br/>predicted: ${(value as number[])[1].toFixed(2)}` },
    legend: { data: ['예측 vs 실제', '완전 예측선'] },
    grid: { left: 60, right: 20, top: 30, bottom: 40 },
    xAxis: { type: 'value', name: 'Actual', scale: true },
    yAxis: { type: 'value', name: 'Predicted', scale: true },
    series: [
      { name: '예측 vs 실제', type: 'scatter', symbolSize: 6, data: points, itemStyle: { color: '#0f172a' } },
      { name: '완전 예측선', type: 'line', data: [[lo, lo], [hi, hi]], lineStyle: { type: 'dashed', color: '#dc2626' }, showSymbol: false },
    ],
  }
})
</script>

<template>
  <v-chart class="w-full h-72" :option="option" autoresize />
</template>
