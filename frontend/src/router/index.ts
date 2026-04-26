import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  { path: '/', redirect: '/upload' },
  { path: '/upload', name: 'upload', component: () => import('@/views/Upload.vue') },
  { path: '/train', name: 'train', component: () => import('@/views/Train.vue') },
  { path: '/evaluate', name: 'evaluate', component: () => import('@/views/Evaluate.vue') },
  { path: '/compare', name: 'compare', component: () => import('@/views/Compare.vue') },
  { path: '/predict', name: 'predict', component: () => import('@/views/Predict.vue') },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
