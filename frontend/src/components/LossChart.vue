<script setup lang="ts">
import { computed, defineComponent } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
  TitleComponent,
} from 'echarts/components'
import type { EChartsOption } from 'echarts'

use([CanvasRenderer, LineChart, GridComponent, TooltipComponent, LegendComponent, TitleComponent])

defineComponent({ components: { VChart } })

const props = defineProps<{
  history: { epoch: number; train_loss: number; val_loss: number }[]
}>()

const option = computed<EChartsOption>(() => ({
  tooltip: { trigger: 'axis' },
  legend: { data: ['Train', 'Validation'] },
  grid: { left: 50, right: 20, top: 30, bottom: 30 },
  xAxis: {
    type: 'category',
    data: props.history.map((h) => h.epoch),
    name: 'epoch',
  },
  yAxis: { type: 'value', name: 'loss' },
  series: [
    {
      name: 'Train',
      type: 'line',
      smooth: true,
      data: props.history.map((h) => Number(h.train_loss.toFixed(6))),
    },
    {
      name: 'Validation',
      type: 'line',
      smooth: true,
      data: props.history.map((h) => Number(h.val_loss.toFixed(6))),
    },
  ],
}))
</script>

<template>
  <v-chart class="w-full h-72" :option="option" autoresize />
</template>
