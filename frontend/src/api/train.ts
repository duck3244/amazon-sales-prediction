import { api } from './client'
import type { components } from './schema'

export type TrainRequest = components['schemas']['TrainRequest']
export type TrainStartResponse = components['schemas']['TrainStartResponse']
export type TrainStatusResponse = components['schemas']['TrainStatusResponse']
export type PreprocessResponse = components['schemas']['PreprocessResponse']
export type JobLogResponse = components['schemas']['JobLogResponse']

export const startTraining = (req: TrainRequest) =>
  api.post<TrainStartResponse>('/train', req).then((r) => r.data)

export const getJob = (jobId: string) =>
  api.get<TrainStatusResponse>(`/train/${jobId}`).then((r) => r.data)

export const getActiveJob = () =>
  api.get<TrainStatusResponse | null>('/train/active').then((r) => r.data)

export const runPreprocess = () =>
  api.post<PreprocessResponse>('/preprocess').then((r) => r.data)

export const getJobLog = (jobId: string, since: number) =>
  api
    .get<JobLogResponse>(`/train/${jobId}/log`, { params: { since } })
    .then((r) => r.data)
