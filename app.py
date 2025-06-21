import gradio as gr
import PyPDF2
import io
import re
import json
import random
from typing import List, Dict, Tuple, Optional
import requests
import time
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# Download required NLTK data with proper error handling
def download_nltk_data():
    """Download required NLTK data with fallback handling"""
    try:
        # Try to find punkt_tab first (newer NLTK versions)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                nltk.download('punkt_tab')
            except Exception:
                # Fallback to punkt for older versions
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt')
        
        # Download stopwords
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
    except Exception as e:
        print(f"Warning: Could not download NLTK data: {e}")
        print("Some features may not work properly.")

# Initialize NLTK data
download_nltk_data()

class DocumentAnalyzer:
    def __init__(self):
        self.document_content = ""
        self.document_sentences = []
        self.conversation_history = []
        self.word_vectors = {}  # Simple word frequency vectors
        self.vocab = []
        self.vocab_dict = {}
        self.sentence_vectors = []
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            with open(pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip()
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    
    def extract_text_from_txt(self, txt_file) -> str:
        """Extract text from uploaded TXT file"""
        try:
            with open(txt_file, 'r', encoding='utf-8') as file:
                content = file.read()
                return content.strip()
        except UnicodeDecodeError:
            try:
                with open(txt_file, 'r', encoding='latin-1') as file:
                    content = file.read()
                    return content.strip()
            except Exception as e:
                return f"Error reading TXT file: {str(e)}"
        except Exception as e:
            return f"Error reading TXT: {str(e)}"
    
    def safe_sent_tokenize(self, text: str) -> List[str]:
        """Safe sentence tokenization with fallback"""
        try:
            return sent_tokenize(text)
        except Exception:
            # Fallback: simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def safe_word_tokenize(self, text: str) -> List[str]:
        """Safe word tokenization with fallback"""
        try:
            return word_tokenize(text)
        except Exception:
            # Fallback: simple word splitting
            return re.findall(r'\b\w+\b', text.lower())
    
    def get_stopwords(self) -> set:
        """Get stopwords with fallback"""
        try:
            return set(stopwords.words('english'))
        except Exception:
            # Fallback: basic English stopwords
            return {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
                'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once'
            }
    
    def preprocess_document(self, text: str):
        """Preprocess document and create simple word vectors"""
        self.document_content = text
        # Split into sentences for better retrieval
        self.document_sentences = self.safe_sent_tokenize(text)
        
        # Create simple word frequency vectors for each sentence
        self.create_word_vectors()
        
    def create_word_vectors(self):
        """Create simple TF-IDF like vectors for semantic search"""
        stop_words = self.get_stopwords()
        
        # Get all unique words across all sentences
        all_words = set()
        sentence_words = []
        
        for sentence in self.document_sentences:
            words = [word.lower() for word in self.safe_word_tokenize(sentence) 
                    if word.isalpha() and word.lower() not in stop_words]
            sentence_words.append(words)
            all_words.update(words)
        
        # Create vocabulary
        self.vocab = list(all_words)
        self.vocab_dict = {word: i for i, word in enumerate(self.vocab)}
        
        # Create vectors for each sentence
        self.sentence_vectors = []
        for words in sentence_words:
            vector = [0] * len(self.vocab)
            word_count = Counter(words)
            for word, count in word_count.items():
                if word in self.vocab_dict:
                    vector[self.vocab_dict[word]] = count
            self.sentence_vectors.append(vector)
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0
            
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def query_to_vector(self, query: str):
        """Convert query to vector using same vocabulary"""
        stop_words = self.get_stopwords()
        words = [word.lower() for word in self.safe_word_tokenize(query) 
                if word.isalpha() and word.lower() not in stop_words]
        
        if not self.vocab:
            return []
            
        vector = [0] * len(self.vocab)
        word_count = Counter(words)
        
        for word, count in word_count.items():
            if word in self.vocab_dict:
                vector[self.vocab_dict[word]] = count
        
        return vector
    
    def generate_summary(self, text: str, max_words: int = 150) -> str:
        """Generate document summary using Hugging Face free API with fallback"""
        try:
            # Use Hugging Face Inference API (free tier)
            API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
            
            # Truncate text if too long
            max_input_length = 1024
            if len(text) > max_input_length:
                text = text[:max_input_length]
            
            payload = {
                "inputs": text,
                "parameters": {
                    "max_length": max_words,
                    "min_length": 50,
                    "do_sample": False
                }
            }
            
            response = requests.post(API_URL, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    summary = result[0].get('summary_text', '')
                    if summary:
                        return summary
            
            # Fallback: Simple extractive summary
            return self.extractive_summary(text, max_words)
            
        except Exception as e:
            print(f"Summary generation error: {e}")
            return self.extractive_summary(text, max_words)
    
    def extractive_summary(self, text: str, max_words: int = 150) -> str:
        """Fallback extractive summary method"""
        sentences = self.safe_sent_tokenize(text)
        if len(sentences) <= 3:
            words = text.split()
            if len(words) <= max_words:
                return text
            return " ".join(words[:max_words]) + "..."
        
        # Simple scoring based on sentence position, length, and keyword frequency
        scored_sentences = []
        
        # Get important keywords from the document
        stop_words = self.get_stopwords()
        words = [word.lower() for word in self.safe_word_tokenize(text) 
                if word.isalpha() and len(word) > 3 and word.lower() not in stop_words]
        word_freq = Counter(words)
        important_words = set([word for word, freq in word_freq.most_common(10)])
        
        for i, sentence in enumerate(sentences[:15]):  # First 15 sentences
            if not sentence.strip():
                continue
                
            score = 1.0 / (i + 1)  # Earlier sentences get higher scores
            
            # Boost score based on sentence length (prefer substantial sentences)
            sentence_words = sentence.split()
            if 10 <= len(sentence_words) <= 30:
                score *= 1.3
            
            # Boost score based on important keywords
            sentence_lower = sentence.lower()
            keyword_count = sum(1 for word in important_words if word in sentence_lower)
            score *= (1 + keyword_count * 0.2)
            
            scored_sentences.append((score, sentence))
        
        if not scored_sentences:
            return "Unable to generate summary from the provided text."
        
        # Sort by score and take top sentences
        scored_sentences.sort(reverse=True)
        summary_sentences = [sent for _, sent in scored_sentences[:4]]
        
        # Reorder sentences by their original position for better flow
        original_positions = []
        for summary_sent in summary_sentences:
            for i, orig_sent in enumerate(sentences):
                if summary_sent == orig_sent:
                    original_positions.append((i, summary_sent))
                    break
        
        original_positions.sort()
        summary = " ".join([sent for _, sent in original_positions])
        
        # Truncate to word limit
        words = summary.split()
        if len(words) > max_words:
            summary = " ".join(words[:max_words]) + "..."
        
        return summary if summary.strip() else "Unable to generate summary."
    
    def find_relevant_context(self, query: str, top_k: int = 3) -> List[Tuple[str, float, int]]:
        """Find most relevant sentences for a query using simple similarity"""
        if not hasattr(self, 'sentence_vectors') or not self.sentence_vectors:
            return []
        
        query_vector = self.query_to_vector(query)
        if not query_vector:
            return []
        
        # Calculate similarities
        similarities = []
        for i, sent_vector in enumerate(self.sentence_vectors):
            if sent_vector:  # Check if vector is not empty
                similarity = self.cosine_similarity(query_vector, sent_vector)
                similarities.append((similarity, i))
        
        if not similarities:
            return []
        
        # Sort by similarity and get top-k
        similarities.sort(reverse=True)
        
        results = []
        for similarity, idx in similarities[:top_k]:
            if similarity > 0.1:  # Threshold for relevance
                if idx < len(self.document_sentences):
                    results.append((self.document_sentences[idx], similarity, idx))
        
        return results
    
    def generate_answer(self, question: str) -> Dict[str, str]:
        """Generate answer to user question"""
        if not self.document_content:
            return {
                "answer": "Please upload a document first.",
                "justification": "",
                "highlighted_text": ""
            }
        
        # Find relevant context
        relevant_contexts = self.find_relevant_context(question, top_k=3)
        
        if not relevant_contexts:
            return {
                "answer": "I couldn't find relevant information in the document to answer this question.",
                "justification": "No sufficiently relevant content found.",
                "highlighted_text": ""
            }
        
        # Combine relevant contexts
        context_text = " ".join([ctx[0] for ctx in relevant_contexts])
        best_context = relevant_contexts[0]
        
        # Try Hugging Face QA model first, then fallback
        try:
            answer = self.query_qa_model(question, context_text)
            if answer == "Unable to generate answer":
                answer = self.simple_answer_extraction(question, context_text)
        except Exception as e:
            print(f"QA model error: {e}")
            answer = self.simple_answer_extraction(question, context_text)
        
        justification = f"This answer is based on content from the document (similarity score: {best_context[1]:.2f}). Reference: Sentence {best_context[2] + 1}"
        
        # Add to conversation history
        self.conversation_history.append({
            "question": question,
            "answer": answer,
            "context": context_text,
            "timestamp": time.time()
        })
        
        return {
            "answer": answer,
            "justification": justification,
            "highlighted_text": best_context[0]
        }
    
    def query_qa_model(self, question: str, context: str) -> str:
        """Query Hugging Face QA model with timeout and error handling"""
        try:
            API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
            
            payload = {
                "inputs": {
                    "question": question,
                    "context": context[:512]  # Limit context length
                }
            }
            
            response = requests.post(API_URL, json=payload, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, dict) and 'answer' in result and result.get('score', 0) > 0.1:
                    return result['answer']
            
            return "Unable to generate answer"
            
        except Exception as e:
            print(f"QA API error: {e}")
            return "Unable to generate answer"
    
    def simple_answer_extraction(self, question: str, context: str) -> str:
        """Simple rule-based answer extraction"""
        question_lower = question.lower()
        context_sentences = self.safe_sent_tokenize(context)
        
        if not context_sentences:
            return "Unable to extract answer from context."
        
        # Extract keywords from question
        stop_words = self.get_stopwords()
        question_words = [word for word in self.safe_word_tokenize(question_lower) 
                         if word.isalpha() and word not in stop_words and len(word) > 2]
        
        # Score sentences based on keyword matches
        best_sentence = ""
        best_score = 0
        
        for sentence in context_sentences:
            if not sentence.strip():
                continue
                
            sentence_lower = sentence.lower()
            
            # Count keyword matches
            matches = sum(1 for word in question_words if word in sentence_lower)
            
            # Boost score for complete phrases
            if len(question_words) >= 2:
                for i in range(len(question_words) - 1):
                    phrase = f"{question_words[i]} {question_words[i+1]}"
                    if phrase in sentence_lower:
                        matches += 2
            
            # Prefer sentences of reasonable length
            sentence_words = sentence.split()
            if 5 <= len(sentence_words) <= 50:
                matches *= 1.2
            
            if matches > best_score:
                best_score = matches
                best_sentence = sentence
        
        if best_sentence:
            return best_sentence
        
        # Return first sentence if no good match
        return context_sentences[0] if context_sentences else "Unable to extract answer from context."
    
    def generate_challenge_questions(self) -> List[Dict[str, str]]:
        """Generate challenge questions from the document"""
        if not self.document_content:
            return []
        
        sentences = self.document_sentences
        if len(sentences) < 5:
            return []
        
        questions = []
        
        # Extract key terms from document
        key_terms = self.extract_key_terms()
        
        if len(key_terms) < 3:
            return []
        
        # Generate different types of questions
        question_templates = [
            ("comprehension", "Based on the document, what is the main point discussed about {}?"),
            ("inference", "What can be inferred about {} from the information provided in the document?"),
            ("analysis", "According to the document, how does {} relate to the overall topic?"),
            ("evaluation", "What evidence does the document provide to support claims about {}?")
        ]
        
        for i, (q_type, template) in enumerate(question_templates[:3]):
            if i < len(key_terms):
                question_text = template.format(key_terms[i])
                
                # Find reference context
                relevant_contexts = self.find_relevant_context(key_terms[i], top_k=1)
                reference = relevant_contexts[0][0] if relevant_contexts else "General document content"
                
                questions.append({
                    "question": question_text,
                    "type": q_type,
                    "reference_context": reference,
                    "key_term": key_terms[i]
                })
        
        return questions
    
    def extract_key_terms(self) -> List[str]:
        """Extract key terms from document using frequency analysis"""
        stop_words = self.get_stopwords()
        
        # Extended stop words for better filtering
        extended_stop_words = stop_words.union({
            'would', 'could', 'should', 'might', 'must', 'shall', 'will',
            'one', 'two', 'three', 'first', 'second', 'third', 'also',
            'however', 'therefore', 'thus', 'hence', 'furthermore',
            'moreover', 'nevertheless', 'nonetheless', 'although'
        })
        
        words = self.safe_word_tokenize(self.document_content.lower())
        words = [word for word in words if word.isalpha() and len(word) > 4 
                and word not in extended_stop_words]
        
        if not words:
            return []
        
        # Get word frequency
        word_freq = Counter(words)
        
        # Filter words that appear at least twice but not too frequently
        total_words = len(words)
        filtered_words = []
        
        for word, freq in word_freq.items():
            # Include words that appear multiple times but not overly common
            if 2 <= freq <= max(3, total_words // 20):
                filtered_words.append((word, freq))
        
        # Sort by frequency and return top terms
        sorted_words = sorted(filtered_words, key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:15]]
    
    def evaluate_user_answer(self, question: str, user_answer: str, reference_context: str) -> Dict[str, str]:
        """Evaluate user's answer to challenge question"""
        if not user_answer.strip():
            return {
                "feedback": "Please provide an answer to evaluate.",
                "score": "0/10",
                "justification": ""
            }
        
        # Find relevant context for evaluation
        relevant_contexts = self.find_relevant_context(user_answer, top_k=2)
        
        if not relevant_contexts:
            score = 3
            feedback = "Your answer doesn't seem to align well with the document content."
        else:
            # Evaluate based on similarity and content quality
            best_similarity = relevant_contexts[0][1]
            
            # Additional checks for answer quality
            answer_length = len(user_answer.split())
            has_specific_details = any(term in user_answer.lower() 
                                    for term in self.extract_key_terms()[:5])
            
            # Scoring logic
            if best_similarity > 0.6:
                base_score = 8
            elif best_similarity > 0.4:
                base_score = 6
            elif best_similarity > 0.2:
                base_score = 4
            else:
                base_score = 2
            
            # Bonus points for detailed answers
            if answer_length >= 20:
                base_score += 1
            
            if has_specific_details:
                base_score += 1
            
            score = min(10, base_score)
            
            # Generate feedback
            if score >= 8:
                feedback = "Excellent! Your answer demonstrates strong understanding and aligns well with the document."
            elif score >= 6:
                feedback = "Good answer! Your response shows solid understanding with good accuracy."
            elif score >= 4:
                feedback = "Fair answer. Your response has some relevance but could be more detailed or accurate."
            else:
                feedback = "Your answer needs improvement. Try to focus more on the specific content from the document."
        
        justification = f"Evaluation based on document content: '{reference_context[:200]}...'"
        
        return {
            "feedback": feedback,
            "score": f"{score}/10",
            "justification": justification
        }

# Initialize the analyzer
analyzer = DocumentAnalyzer()
current_challenge_questions = []

def process_document(file):
    """Process uploaded document"""
    if file is None:
        return "Please upload a document.", "", ""
    
    try:
        if file.name.endswith('.pdf'):
            text = analyzer.extract_text_from_pdf(file.name)
        elif file.name.endswith('.txt'):
            text = analyzer.extract_text_from_txt(file.name)
        else:
            return "Please upload a PDF or TXT file.", "", ""
        
        if text.startswith("Error"):
            return text, "", ""
        
        if len(text.strip()) < 100:
            return "Document seems too short or empty. Please upload a document with substantial content.", "", ""
        
        # Preprocess document
        analyzer.preprocess_document(text)
        
        # Generate summary
        summary = analyzer.generate_summary(text)
        
        return f"✅ Document processed successfully! ({len(text)} characters, {len(analyzer.document_sentences)} sentences)", summary, "📚 Document is ready for questions and challenges!"
        
    except Exception as e:
        return f"❌ Error processing document: {str(e)}", "", ""

def ask_question(question):
    """Handle user questions"""
    if not question.strip():
        return "Please enter a question.", "", ""
    
    try:
        result = analyzer.generate_answer(question)
        return result["answer"], result["justification"], result["highlighted_text"]
    except Exception as e:
        return f"Error generating answer: {str(e)}", "", ""

def generate_challenges():
    """Generate challenge questions"""
    global current_challenge_questions
    
    if not analyzer.document_content:
        return "Please upload a document first.", "", "", ""
    
    try:
        current_challenge_questions = analyzer.generate_challenge_questions()
        
        if not current_challenge_questions:
            return "Unable to generate meaningful questions from this document. The document might be too short or lack sufficient content.", "", "", ""
        
        questions_text = "\n\n".join([
            f"**Question {i+1}:** {q['question']}\n*Type: {q['type'].title()}*"
            for i, q in enumerate(current_challenge_questions)
        ])
        
        return (questions_text, 
               "Select a question number (1-3) and provide your detailed answer below:",
               "",
               f"✅ {len(current_challenge_questions)} challenge questions generated!")
    except Exception as e:
        return f"Error generating challenges: {str(e)}", "", "", ""

def evaluate_challenge_answer(question_num, user_answer):
    """Evaluate user's challenge answer"""
    try:
        if not question_num.strip():
            return "Please select a question number (1-3).", ""
            
        q_idx = int(question_num) - 1
        if q_idx < 0 or q_idx >= len(current_challenge_questions):
            return "Please select a valid question number (1-3).", ""
        
        question_data = current_challenge_questions[q_idx]
        result = analyzer.evaluate_user_answer(
            question_data["question"],
            user_answer,
            question_data["reference_context"]
        )
        
        feedback = f"**Score: {result['score']}**\n\n{result['feedback']}"
        justification = result['justification']
        
        return feedback, justification
        
    except ValueError:
        return "Please enter a valid question number (1, 2, or 3).", ""
    except Exception as e:
        return f"Error evaluating answer: {str(e)}", ""

def show_history():
    """Display conversation history"""
    if not analyzer.conversation_history:
        return "No conversation history yet. Start asking questions!"
    
    try:
        history = []
        for i, conv in enumerate(analyzer.conversation_history[-10:], 1):  # Last 10 conversations
            history.append(f"**Q{i}:** {conv['question']}")
            history.append(f"**A{i}:** {conv['answer'][:300]}{'...' if len(conv['answer']) > 300 else ''}")
            history.append("---")
        
        return "\n\n".join(history)
    except Exception as e:
        return f"Error displaying history: {str(e)}"

# Create Gradio interface with improved styling
with gr.Blocks(
    title="📚 Document Analysis Assistant", 
    theme=gr.themes.Soft(),
    css=".gradio-container {font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}"
) as app:
    
    gr.Markdown("""
    # 📚 Advanced Document Analysis Assistant
    
    Upload your PDF or TXT document and interact with it intelligently:
    - **📄 Auto Summary**: Get instant document overview
    - **❓ Ask Anything**: Natural language Q&A with justifications  
    - **🎯 Challenge Me**: Test comprehension with AI-generated questions
    - **💬 Memory**: Maintains conversation context
    
    *Built with free AI models - no API keys required!*
    """)
    
    with gr.Tab("📄 Document Upload"):
        gr.Markdown("### Upload your document to get started")
        
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="📁 Upload Document (PDF/TXT)",
                    file_types=[".pdf", ".txt"],
                    type="filepath"
                )
                upload_btn = gr.Button("🚀 Process Document", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                upload_status = gr.Textbox(label="📋 Status", interactive=False)
                doc_summary = gr.Textbox(
                    label="📄 Auto Summary (≤150 words)",
                    lines=8,
                    interactive=False,
                    placeholder="Document summary will appear here..."
                )
        
        processing_status = gr.Textbox(label="ℹ️ Processing Info", interactive=False)
    
    with gr.Tab("❓ Ask Anything"):
        gr.Markdown("### 🤔 Ask questions about your document")
        
        with gr.Row():
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label="🔍 Your Question",
                    placeholder="What is the main conclusion? How does X relate to Y?",
                    lines=3
                )
                ask_btn = gr.Button("💡 Get Answer", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                gr.Markdown("""
                **💡 Example Questions:**
                - What are the key findings?
                - How does the author support their argument?
                - What is the relationship between X and Y?
                - What evidence is provided for Z?
                - What are the main recommendations?
                """)
        
        with gr.Row():
            with gr.Column():
                answer_output = gr.Textbox(label="✅ Answer", lines=6, interactive=False)
                justification_output = gr.Textbox(label="🔍 Justification & Reference", lines=3, interactive=False)
            with gr.Column():
                highlighted_text = gr.Textbox(label="📖 Supporting Text from Document", lines=6, interactive=False)
    
    with gr.Tab("🎯 Challenge Me"):
        gr.Markdown("### 🧠 Test your understanding with AI-generated questions")
        
        with gr.Row():
            generate_btn = gr.Button("🎲 Generate Challenge Questions", variant="primary", size="lg")
            challenge_status = gr.Textbox(label="📋 Status", interactive=False)
        
        challenge_questions = gr.Textbox(
            label="📝 Challenge Questions",
            lines=10,
            interactive=False,
            placeholder="Challenge questions will appear here after generation..."
        )
        
        gr.Markdown("### ✍️ Answer Section")
        with gr.Row():
            with gr.Column():
                question_selector = gr.Textbox(
                    label="🔢 Question Number (1-3)",
                    placeholder="Enter: 1, 2, or 3",
                    max_lines=1
                )
                user_answer_input = gr.Textbox(
                    label="✍️ Your Answer",
                    placeholder="Provide a detailed answer based on the document...",
                    lines=5
                )
                evaluate_btn = gr.Button("📊 Evaluate My Answer", variant="secondary", size="lg")
            
            with gr.Column():
                gr.Markdown("""
                **📚 Tips for Better Scores:**
                - Reference specific details from the document
                - Provide comprehensive explanations
                - Use evidence-based reasoning
                - Write at least 2-3 sentences
                - Stay focused on the document content
                """)
        
        with gr.Row():
            evaluation_feedback = gr.Textbox(label="📊 Evaluation & Score", lines=4, interactive=False)
            evaluation_justification = gr.Textbox(label="🔍 Evaluation Details", lines=4, interactive=False)
        
        challenge_instructions = gr.Textbox(
            label="📋 Instructions",
            interactive=False,
            placeholder="Generate questions first, then select a question number and provide your answer."
        )
    
    with gr.Tab("💬 Conversation History"):
        gr.Markdown("### 📚 Your Q&A History")
        
        history_btn = gr.Button("🔄 Refresh History", variant="secondary")
        conversation_history = gr.Textbox(
            label="💬 Recent Conversations",
            lines=15,
            interactive=False,
            placeholder="Your conversation history will appear here..."
        )
        
        gr.Markdown("*Shows your last 10 questions and answers*")
    
    with gr.Tab("ℹ️ Help & About"):
        gr.Markdown("""
        ## 🚀 How to Use This Tool
        
        ### 1. **📄 Upload Document**
        - Support formats: PDF, TXT
        - Minimum 100 characters for meaningful analysis
        - Get instant auto-summary after upload
        
        ### 2. **❓ Ask Questions**
        - Natural language queries about document content
        - Get AI-powered answers with justifications
        - See highlighted supporting text from document
        
        ### 3. **🎯 Take Challenges**
        - AI generates comprehension questions
        - Test your understanding with different question types
        - Get scored feedback (0-10 scale)
        
        ### 4. **💬 Track History**
        - View your recent Q&A conversations
        - Reference previous answers and questions
        
        ---
        
        ## 🔧 Technical Features
        
        - **Free AI Models**: Uses Hugging Face free inference APIs
        - **Fallback Systems**: Robust error handling with local processing
        - **Smart Search**: Semantic similarity matching for relevant content
        - **Multiple Question Types**: Comprehension, inference, analysis, evaluation
        - **Context-Aware**: Maintains conversation memory
        
        ---
        
        ## 💡 Tips for Best Results
        
        **For Documents:**
        - Use well-structured, substantive content
        - Ensure documents are clearly readable
        - Longer documents (500+ words) work better
        
        **For Questions:**
        - Be specific and clear
        - Ask about content actually in the document
        - Use complete sentences
        
        **For Challenges:**
        - Read the document thoroughly first
        - Provide detailed, evidence-based answers
        - Reference specific information from the text
        
        ---
        
        ## ⚠️ Limitations
        
        - Depends on free AI service availability
        - Best with English documents
        - Processing time varies with document length
        - Internet connection required for AI features
        
        ---
        
        **Built with ❤️ using Gradio, NLTK, and Hugging Face**
        """)

    # Event handlers
    upload_btn.click(
        fn=process_document,
        inputs=[file_input],
        outputs=[upload_status, doc_summary, processing_status]
    )
    
    ask_btn.click(
        fn=ask_question,
        inputs=[question_input],
        outputs=[answer_output, justification_output, highlighted_text]
    )
    
    # Allow Enter key to submit questions
    question_input.submit(
        fn=ask_question,
        inputs=[question_input],
        outputs=[answer_output, justification_output, highlighted_text]
    )
    
    generate_btn.click(
        fn=generate_challenges,
        outputs=[challenge_questions, challenge_instructions, evaluation_feedback, challenge_status]
    )
    
    evaluate_btn.click(
        fn=evaluate_challenge_answer,
        inputs=[question_selector, user_answer_input],
        outputs=[evaluation_feedback, evaluation_justification]
    )
    
    history_btn.click(
        fn=show_history,
        outputs=[conversation_history]
    )
    
    # Auto-refresh history when switching to history tab
    # Note: This is a simplified approach - in practice, you might want to use gr.State for better state management

# Launch the application
if __name__ == "__main__":
    print("🚀 Starting Document Analysis Assistant...")
    print("📋 Features: PDF/TXT processing, Q&A, Challenge questions, Conversation history")
    print("🔗 Using free Hugging Face models with local fallbacks")
    
    app.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True to create public link
        debug=False,            # Set to True for development
        show_error=True,        # Show detailed error messages
        inbrowser=True          # Auto-open in browser
    )
