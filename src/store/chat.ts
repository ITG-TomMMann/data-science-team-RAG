import { create } from 'zustand';
import { Message } from '../types/chat';

type ChatStore = {
  messages: Message[];
  isTyping: boolean;
  selectedFolder: string;
  addMessage: (message: Omit<Message, 'id'>) => void;
  setTyping: (isTyping: boolean) => void;
  setSelectedFolder: (folder: string) => void;
};

export const useChatStore = create<ChatStore>((set) => ({
  messages: [],
  isTyping: false,
  selectedFolder: 'itg',
  addMessage: (message) =>
    set((state) => ({
      messages: [
        ...state.messages,
        { ...message, id: Math.random().toString(36).substr(2, 9) },
      ],
    })),
  setTyping: (isTyping) => set({ isTyping }),
  setSelectedFolder: (folder) => set({ selectedFolder: folder }),
}));
