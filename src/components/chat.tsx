import { Send } from 'lucide-react';
import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { useAuthStore } from '@/lib/store';
import { ChevronDown, ChevronUp, FileText } from 'lucide-react';

const FOLDERS = [
  'ITG hypotheses',
  "Tom's analysis",
  'Accenture AB tests',
  'Everything combined',
] as const;

type Message = {
  role: 'user' | 'assistant';
  content: string;
  sources?: {
    content?: string;
    doc_id: string;
    page_number: number;
    score?: number;
    imageUrl?: string; // Add property to store image URL
  }[];
};

const QUICK_LINKS = [
  {
    title: 'Mobile Content Engagement on US Range Rover Nameplate Pages',
    href: '#mobile-content',
  },
  {
    title: 'Nameplate Visualiser Model Selection',
    href: '#nameplate-visualiser',
  },
  {
    title: 'US Forms Copy Optimization',
    href: '#forms-copy',
  },
  {
    title: 'Hybrid Vehicle Search Trends in the US',
    href: '#hybrid-trends',
  },
];

export function Chat() {
  const navigate = useNavigate();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [selectedFolder, setSelectedFolder] = useState<typeof FOLDERS[number]>('ITG hypotheses');
  const [isLoading, setIsLoading] = useState(false);
  const [expandedSources, setExpandedSources] = useState<Record<string, boolean>>({});
  const [loadingImages, setLoadingImages] = useState<Record<string, boolean>>({});
  const { logout, getToken } = useAuthStore();
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);
  
  // Function to fetch document image from GCS via backend
  const fetchDocumentImage = async (docId: string, pageNumber: number) => {
    const sourceKey = `${docId}-${pageNumber}`;
    
    setLoadingImages(prev => ({ ...prev, [sourceKey]: true }));
    
    try {
      const token = typeof getToken === 'function' ? getToken() : null;
      const baseUrl = `${window.location.protocol}//${window.location.hostname}:8000`;
      
      // Make request to backend to get the document page as an image
      const response = await fetch(`${baseUrl}/document-page`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token && { 'Authorization': `Bearer ${token}` }),
        },
        body: JSON.stringify({
          doc_id: docId,
          page_number: pageNumber,
          folder: selectedFolder,
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status} ${response.statusText}`);
      }
      
      const result = await response.json();
      
      if (result.image_url) {
        // Update the message source with the image URL
        setMessages(prev => 
          prev.map(message => {
            if (message.role === 'assistant' && message.sources) {
              return {
                ...message,
                sources: message.sources.map(source => {
                  if (source.doc_id === docId && source.page_number === pageNumber) {
                    return { ...source, imageUrl: result.image_url };
                  }
                  return source;
                }),
              };
            }
            return message;
          })
        );
      }
    } catch (error) {
      console.error(`Error fetching image for ${docId} page ${pageNumber}:`, error);
    } finally {
      setLoadingImages(prev => ({ ...prev, [sourceKey]: false }));
    }
  };
  
  // Toggle expanded state for source images
  const toggleSourceExpanded = (docId: string, pageNumber: number) => {
    const key = `${docId}-${pageNumber}`;
    setExpandedSources(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      // Fix: Remove /api prefix to match backend expectation
      const baseUrl = `${window.location.protocol}//${window.location.hostname}:8000`;
      // Get auth token if available
      const token = typeof getToken === 'function' ? getToken() : null;
      
      const response = await fetch(`${baseUrl}/query`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept-Language': 'en-US,en', // Add language preference
          ...(token && { 'Authorization': `Bearer ${token}` }), // Add auth token if available
        },
        body: JSON.stringify({ 
          query: userMessage,
          language: 'en', // Explicitly request English
          folder: selectedFolder, // Include the selected folder/context
          include_sources: true // Ensure sources are returned
        }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log('Response data:', data); // Debug log
      
      // Log detailed information about sources
      if (data.sources && data.sources.length > 0) {
        console.log(`Found ${data.sources.length} sources`);
        
        // Create a new message object with the response data
        const newMessage: Message = { 
          role: 'assistant', 
          content: data.answer || 'Sorry, no answer was provided.', 
          sources: data.sources || [] 
        };
        
        setMessages(prev => [...prev, newMessage]);
        
        // Fetch image URLs for each source
        data.sources.forEach(source => {
          fetchDocumentImage(source.doc_id, source.page_number);
        });
      } else {
        console.warn('No sources found in the response');
        console.log('Full response:', JSON.stringify(data));
        
        setMessages(prev => [
          ...prev,
          { role: 'assistant', content: data.answer || 'Sorry, no answer was provided.' }
        ]);
      }
      // Message setting is now handled in the conditional block above
    } catch (error) {
      console.error('Error:', error);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Sorry, there was an error: ${error instanceof Error ? error.message : 'Unknown error'}` },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  return (
    <div className="flex h-screen flex-col">
      {/* Notices */}
      <div className="space-y-1 text-center text-sm">
        <div className="bg-gray-100 px-4 py-2 text-gray-700">
          Data collection notice: All conversations are stored for analysis purposes
        </div>
        <div className="bg-gray-50 px-4 py-2 text-gray-600">
          Note: This knowledge assistant can make mistakes. Please verify important information.
        </div>
      </div>

      {/* Header with folder selection and logout */}
      <div className="border-b px-4 py-2 flex justify-between items-center">
        <select
          value={selectedFolder}
          onChange={(e) => setSelectedFolder(e.target.value as typeof FOLDERS[number])}
          className="rounded-md border-gray-300 py-2 pl-3 pr-10 text-sm focus:border-[#e7298a] focus:outline-none focus:ring-[#e7298a]"
        >
          {FOLDERS.map((folder) => (
            <option key={folder} value={folder}>
              {folder}
            </option>
          ))}
        </select>
        <Button variant="outline" onClick={handleLogout} size="sm">
          Logout
        </Button>
      </div>

      {/* Chat messages */}
      <div 
        ref={chatContainerRef}
        className="flex-1 overflow-y-auto p-4 space-y-4 max-w-3xl mx-auto w-full"
      >
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full space-y-8">
            <h1 className="text-2xl font-semibold text-gray-700">How can I help you?</h1>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-2xl">
              {QUICK_LINKS.map((link) => (
                <a
                  key={link.href}
                  href={link.href}
                  className="p-4 border rounded-lg hover:bg-gray-50 transition-colors duration-200 text-gray-700 hover:text-gray-900"
                >
                  {link.title}
                </a>
              ))}
            </div>
          </div>
        ) : (
          messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${
                message.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              <div
                className={`max-w-[80%] rounded-lg px-4 py-2 ${
                  message.role === 'user'
                    ? 'bg-[#e7298a] text-white'
                    : 'bg-gray-100 text-gray-900'
                }`}
              >
                <p className="whitespace-pre-wrap">{message.content}</p>
                {message.sources && message.sources.length > 0 && (
                  <div className="mt-2 text-sm">
                    <p className="font-semibold">Sources:</p>
                    <div className="space-y-2 mt-1">
                      {message.sources.map((source, i) => {
                        const sourceKey = `${source.doc_id}-${source.page_number}`;
                        const isExpanded = expandedSources[sourceKey];
                        
                        return (
                          <div key={i} className="border rounded-md overflow-hidden">
                            <div 
                              className="flex items-center justify-between p-2 bg-gray-50 cursor-pointer"
                              onClick={() => toggleSourceExpanded(source.doc_id, source.page_number)}
                            >
                              <div className="flex items-center space-x-2">
                                <FileText className="h-4 w-4 text-gray-500" />
                                <span>{source.doc_id}, Page {source.page_number}</span>
                              </div>
                              {isExpanded ? (
                                <ChevronUp className="h-4 w-4 text-gray-500" />
                              ) : (
                                <ChevronDown className="h-4 w-4 text-gray-500" />
                              )}
                            </div>
                            
                            {isExpanded && (
                              <div className="p-2 bg-white">
                                {loadingImages[sourceKey] ? (
                                  <div className="flex justify-center items-center h-32">
                                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-500"></div>
                                  </div>
                                ) : source.imageUrl ? (
                                  <div className="flex justify-center">
                                    <img
                                      src={source.imageUrl}
                                      alt={`${source.doc_id} - Page ${source.page_number}`}
                                      className="max-w-full h-auto border"
                                    />
                                  </div>
                                ) : (
                                  <div className="text-center py-4 text-gray-500">
                                    Image not available
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="flex justify-start">
            <div className="max-w-[80%] rounded-lg bg-gray-100 px-4 py-2">
              <div className="flex space-x-2">
                <div className="h-2 w-2 animate-bounce rounded-full bg-gray-400"></div>
                <div className="h-2 w-2 animate-bounce rounded-full bg-gray-400" style={{ animationDelay: '0.2s' }}></div>
                <div className="h-2 w-2 animate-bounce rounded-full bg-gray-400" style={{ animationDelay: '0.4s' }}></div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input form */}
      <form onSubmit={handleSubmit} className="border-t p-4 max-w-3xl mx-auto w-full">
        <div className="flex space-x-4">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            disabled={isLoading}
            className="flex-1"
          />
          <Button type="submit" disabled={isLoading}>
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </form>
    </div>
  );
}