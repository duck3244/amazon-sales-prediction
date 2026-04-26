import { api } from './client'
import type { components } from './schema'

export type DatasetSummary = components['schemas']['DatasetSummary']
export type UploadResponse = components['schemas']['UploadResponse']

export const uploadDataset = async (file: File): Promise<UploadResponse> => {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post<UploadResponse>('/data/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 5 * 60_000, // 큰 CSV는 시간이 걸림
  })
  return data
}

export const fetchSummary = async (): Promise<DatasetSummary> => {
  const { data } = await api.get<DatasetSummary>('/data/summary')
  return data
}
