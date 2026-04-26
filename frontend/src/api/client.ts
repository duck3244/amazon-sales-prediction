import axios from 'axios'

export const api = axios.create({
  baseURL: '/api',
  timeout: 60_000,
})

api.interceptors.response.use(
  (res) => res,
  (err) => {
    const detail = err?.response?.data?.detail ?? err?.message ?? 'Unknown error'
    return Promise.reject(new Error(typeof detail === 'string' ? detail : JSON.stringify(detail)))
  },
)

export interface HealthResponse {
  status: string
  torch_version: string
  cuda_available: boolean
  cuda_device: string | null
}

export const fetchHealth = () => api.get<HealthResponse>('/health').then((r) => r.data)
