import { api } from './client'
import type { components } from './schema'

export type EvaluationResponse = components['schemas']['EvaluationResponse']
export type TrainedModelInfo = components['schemas']['TrainedModelInfo']
export type SinglePredictRequest = components['schemas']['SinglePredictRequest']
export type SinglePredictResponse = components['schemas']['SinglePredictResponse']
export type CompareResponse = components['schemas']['CompareResponse']
export type CompareEntry = components['schemas']['CompareEntry']

export const listModels = () =>
  api.get<TrainedModelInfo[]>('/models').then((r) => r.data)

export const evaluateBatch = (modelType: string) =>
  api
    .post<EvaluationResponse>('/predict/batch', null, { params: { model_type: modelType } })
    .then((r) => r.data)

export const compareModels = () =>
  api.post<CompareResponse>('/predict/compare').then((r) => r.data)

export const predictSingle = (modelType: string, features: number[]) =>
  api
    .post<SinglePredictResponse>(
      '/predict/single',
      { features },
      { params: { model_type: modelType } },
    )
    .then((r) => r.data)

export interface BatchPredictResult {
  blob: Blob
  filename: string
  rows: number
  elapsedSeconds: number
  model: string
}

export const predictBatch = async (
  modelType: string,
  file: File,
): Promise<BatchPredictResult> => {
  const form = new FormData()
  form.append('file', file)
  const res = await api.post<Blob>('/predict/batch_csv', form, {
    params: { model_type: modelType },
    responseType: 'blob',
    timeout: 5 * 60_000,
  })
  const cd = (res.headers['content-disposition'] as string | undefined) ?? ''
  const match = /filename="([^"]+)"/.exec(cd)
  return {
    blob: res.data,
    filename: match?.[1] ?? `predictions_${modelType}.csv`,
    rows: Number(res.headers['x-prediction-rows'] ?? 0),
    elapsedSeconds: Number(res.headers['x-prediction-elapsed'] ?? 0),
    model: String(res.headers['x-prediction-model'] ?? modelType),
  }
}
