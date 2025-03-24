import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, FolderSearch } from 'lucide-react';
import { useChatStore } from '../store/chat';
import { cn } from '../lib/utils';

const SUGGESTIONS = [
  'Mobile Content Engagement on US Range Rover Nameplate Pages',
  'Nameplate Visualiser Model Selection',
  'US Forms Copy Optimization',
  'Hybrid Vehicle Search Trends in the US',
];

const FOLDERS = [
  { id: 'itg', label: 'ITG hypotheses' },
  { id: 'tom', label: "Tom's analysis" },
  { id: 'accenture', label: 'Accenture AB tests' },
  { id: 'combined', label: 'Everything combined' },
];

export function ChatInterface() {
  const [input, setInput] = useState('');
  const [isInitialView, setIsInitialView] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { messages, isTyping, selectedFolder, addMessage, setTyping, setSelectedFolder } = useChatStore();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const handleSuggestionClick = async (suggestion: string) => {
    setIsInitialView(false);
    setInput('');
    addMessage({ role: 'user', content: suggestion });
    setTyping(true);

    try {
      const response = await fetch('/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: suggestion }),
      });

      const data = await response.json();
      addMessage({
        role: 'assistant',
        content: data.response,
      });
    } catch (error) {
      console.error('Error:', error);
      addMessage({
        role: 'assistant',
        content: 'Sorry, there was an error processing your request.',
      });
    } finally {
      setTyping(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    setIsInitialView(false);
    const query = input;
    setInput('');
    addMessage({ role: 'user', content: query });
    setTyping(true);

    try {
      const response = await fetch('/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      const data = await response.json();
      addMessage({
        role: 'assistant',
        content: data.response,
      });
    } catch (error) {
      console.error('Error:', error);
      addMessage({
        role: 'assistant',
        content: 'Sorry, there was an error processing your request.',
      });
    } finally {
      setTyping(false);
    }
  };

  const showSuggestions = selectedFolder === 'itg' || selectedFolder === 'combined';

  return (
    <div className="flex flex-col h-screen bg-white">
      {/* Notification Banner */}
      <div className="bg-gray-100 p-3 text-sm text-center border-b">
        Data collection notice: All conversations are stored for analysis purposes
      </div>

      {/* Folder Selection */}
      <div className="border-b p-4">
        <div className="flex items-center space-x-4">
          <FolderSearch className="w-5 h-5 text-gray-500" />
          <select
            value={selectedFolder}
            onChange={(e) => setSelectedFolder(e.target.value)}
            className="flex-1 p-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-itg-pink"
          >
            {FOLDERS.map((folder) => (
              <option key={folder.id} value={folder.id}>
                {folder.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-4">
        <AnimatePresence>
          {isInitialView ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="max-w-6xl mx-auto"
            >
              <h1 className="text-3xl font-semibold text-center mb-8">
                What can I help with?
              </h1>
              {showSuggestions && (
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  {SUGGESTIONS.map((suggestion) => (
                    <button
                      key={suggestion}
                      onClick={() => handleSuggestionClick(suggestion)}
                      className="p-4 text-left rounded-lg border hover:border-itg-pink hover:bg-pink-50 transition-colors h-full"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              )}
            </motion.div>
          ) : (
            <div className="max-w-2xl mx-auto space-y-4">
              {messages.map((message) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={cn(
                    'p-4 rounded-lg',
                    message.role === 'user'
                      ? 'bg-itg-pink text-white ml-auto'
                      : 'bg-chat-gray text-gray-900'
                  )}
                >
                  {message.content}
                </motion.div>
              ))}
              {isTyping && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="bg-chat-gray text-gray-900 p-4 rounded-lg"
                >
                  <div className="flex space-x-2">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }} />
                  </div>
                </motion.div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </AnimatePresence>
      </div>

      {/* Input Area */}
      <form onSubmit={handleSubmit} className="border-t p-4">
        <div className="max-w-2xl mx-auto flex items-center space-x-4">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="PROVIDE"
            className="flex-1 p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-itg-pink"
          />
          <button
            type="submit"
            disabled={!input.trim()}
            className="p-2 text-white bg-itg-pink rounded-full hover:bg-itg-pink-dark transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </form>
    </div>
  );
}