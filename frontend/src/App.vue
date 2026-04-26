<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
import HealthBadge from '@/components/HealthBadge.vue'

const navItems = [
  { to: '/upload', label: '데이터 업로드' },
  { to: '/train', label: '모델 학습' },
  { to: '/evaluate', label: '평가' },
  { to: '/compare', label: '모델 비교' },
  { to: '/predict', label: '예측' },
]
</script>

<template>
  <div class="min-h-screen flex flex-col">
    <header class="bg-white border-b border-slate-200">
      <div class="max-w-6xl mx-auto px-6 py-4 flex items-center gap-6">
        <h1 class="text-lg font-bold tracking-tight">
          📦 Amazon Sales Prediction
        </h1>
        <nav class="flex gap-1">
          <RouterLink
            v-for="item in navItems"
            :key="item.to"
            :to="item.to"
            class="px-3 py-1.5 rounded text-sm text-slate-700 hover:bg-slate-100"
            active-class="bg-slate-900 text-white hover:bg-slate-900"
          >
            {{ item.label }}
          </RouterLink>
        </nav>
        <div class="ml-auto">
          <HealthBadge />
        </div>
      </div>
    </header>

    <main class="flex-1 max-w-6xl w-full mx-auto px-6 py-8">
      <RouterView />
    </main>
  </div>
</template>
