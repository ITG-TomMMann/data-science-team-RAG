export type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
};

export type Source = {
  metadata: {
    doc_id: string;
    page_number: number;
    paragraph_index: number;
    folder_path: string;
  };
  content: string;
  context: string;
  score: number;
};

export type QueryResponse = {
  answer: string;
  sources: Source[];
};
