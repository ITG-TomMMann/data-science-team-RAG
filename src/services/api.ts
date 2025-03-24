import axios from 'axios';
import { QueryResponse } from '../types/chat';

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
});

export const queryAPI = async (query: string): Promise<QueryResponse> => {
  const response = await api.post<QueryResponse>('/query', { query });
  return response.data;
};
