import { pipeline, env } from '@xenova/transformers';
import { NLIClassificationPipeline } from './NLI.js';

// Few-shot examples: provide example texts for each category
export const fewShotExamples = {
  'system_configuration': [
    'Check the server status',
    'Is the system running properly?',
    'Monitor the infrastructure health',
    'What is the uptime of the service?',
    "What's the current system health status?",
    'Check the logs for any errors',
    'How many users are currently online?',
    'Is the database alive?',
    'Monitor the CPU and memory usage',
    'Restart the application server'
  ],
  'data_analysis': [
    'Analyze the scores from class L01',
    'Show me statistics about student performance in this course',
    'What are the average grades?',
    'Generate a report on quiz results',
    'Compare performance across different classes',
    'What percentage of students passed the exam?',
    'Show me the grade distribution for this course',
    'Analyze the improvement trend over time',
    'Calculate the median score for the assignment',
    'Generate analytics dashboard for student progress',
    'Explain how students usually struggle with limits and derivatives.'
  ],
  'quiz_creation': [
    'Create a quiz about mathematics',
    'Generate assessment questions for chapter 5',
    'Design a test for chapter 3',
    'Make practice questions about photosynthesis',
    'Design practice problems for calculus',
    'Create exercises for linear algebra',
    'Generate practice problems for students',
    'Make a test about chemistry',
    'Build assessment questions for physics',
    'Make a test about history of Vietnam'
  ]
};

export class FeatureEmbeddedClassifier {
  static embeddingModel = 'Xenova/all-MiniLM-L6-v2';
  static embeddingInstance = null;
  static categoryEmbeddings = null;
  
  // Filter options
  static useNLIFilter = true; // Enable/disable NLI pre-filtering
  static nliConfidenceThreshold = 0.5; // Minimum confidence for "related to education"

  static async getInstance(progress_callback = null) {
    if (this.embeddingInstance === null) {
      // NOTE: Uncomment this to change the cache directory
      // env.cacheDir = './.cache';

      this.embeddingInstance = await pipeline('feature-extraction', this.embeddingModel, { progress_callback });
      
      // Pre-compute embeddings for all examples
      await this.computeCategoryEmbeddings();
    }

    return this.embeddingInstance;
  }

  static async computeCategoryEmbeddings() {
    if (this.categoryEmbeddings !== null) return;

    this.categoryEmbeddings = {};

    // Object.entries returns [key, value] pairs
    // Visualization for categoryEnbeddings structure:
    // {
    //   'system_configuration': [ [embedding1], [embedding2], ... ],
    //   'data_analysis': [ [embedding1], [embedding2], ... ],
    //   ...
    // }
    for (const [category, examples] of Object.entries(fewShotExamples)) {
      // Get embeddings for all examples in this category
      const embeddings = [];
      for (const example of examples) {
          const result = await this.embeddingInstance(example, { pooling: 'mean', normalize: true });
        embeddings.push(result.data);
      }
      this.categoryEmbeddings[category] = embeddings;
    }
  }

  static cosineSimilarity(vecA, vecB) {
    let dotProduct = 0;
    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
    }
    return dotProduct; // Vectors are already normalized
  }

  static async classify(text, options = {}) {
    const { 
      skipFilter = false, // Set to true to bypass NLI filter
      returnFilterInfo = true // Set to true to include filter information in response
    } = options;
    
    // Step 1: NLI Filter (optional, but recommended)
    let filterResult = null;
    if (this.useNLIFilter && !skipFilter) {
      filterResult = await NLIClassificationPipeline.isRelatedToELearning(text);
      
      // If message is unrelated to education, return early
      if (!filterResult.isRelated) {
        const response = {
          sequence: text,
          labels: ['unrelated'],
          scores: [1.0],
          filteredOut: true,
          reason: 'Message not related to e-learning system'
        };
        
        if (returnFilterInfo) {
          response.filterResult = filterResult;
        }
        
        return response;
      }
    }
    
    // Step 2: Feature Embedding Classification (only for education-related messages)
    await this.getInstance();

    // Get embedding for the input text
    const textEmbedding = await this.embeddingInstance(text, { pooling: 'mean', normalize: true });
    const textVector = textEmbedding.data;

    // Calculate similarity with each category (excluding 'unrelated' since it passed NLI filter)
    const categoryScores = {};
    
    // Strategy options:
    // 1. 'max' - Use highest similarity (best for few-shot, finds closest match)
    // 2. 'mean' - Average all similarities (original approach, can dilute strong matches)
    // 3. 'top-k' - Average top K most similar examples (balanced approach)
    const SIMILARITY_STRATEGY = 'top-k'; // Change to 'mean' or 'top-k' to try different approaches
    const TOP_K = 3; // Used when strategy is 'top-k'
    
    // Eg: 'system_configuration': [ [embedding1], [embedding2], ... ],
    for (const [category, embeddings] of Object.entries(this.categoryEmbeddings)) {
      // Skip 'unrelated' category if NLI filter is enabled (we already filtered it out)
      if (this.useNLIFilter && category === 'unrelated') {
        continue;
      }
      
      // Calculate similarity scores with all examples in this category
      const similarities = embeddings.map(exampleEmbedding => 
        this.cosineSimilarity(textVector, exampleEmbedding)
      );
      
      // Choose scoring strategy
      let score;
      switch (SIMILARITY_STRATEGY) {
        case 'max':
          // Use the highest similarity (if ANY example matches well, it's likely this category)
          score = Math.max(...similarities);
          break;
          
        case 'mean':
          // Average all similarities (can be diluted by weak examples)
          score = similarities.reduce((sum, sim) => sum + sim, 0) / similarities.length;
          break;
          
        case 'top-k':
          // Average the top K most similar examples (balanced approach)
          const topK = similarities.sort((a, b) => b - a).slice(0, Math.min(TOP_K, similarities.length));
          score = topK.reduce((sum, sim) => sum + sim, 0) / topK.length;
          break;
          
        default:
          throw new Error(`Unknown similarity strategy: ${SIMILARITY_STRATEGY}`);
      }
      
      categoryScores[category] = score;
    }

    // Sort categories by score
    const sortedCategories = Object.entries(categoryScores)
      .sort((a, b) => b[1] - a[1])
      .map(([category, score]) => ({ category, score }));

    // Normalize scores to sum to 1
    const maxScore = sortedCategories[0].score;
    const minScore = sortedCategories[sortedCategories.length - 1].score;
    const range = maxScore - minScore || 1;

    const normalizedResults = sortedCategories.map(({ category, score }) => ({
      category,
      score: (score - minScore) / range
    }));

    // Re-normalize to sum to 1
    const sum = normalizedResults.reduce((acc, { score }) => acc + score, 0);
    normalizedResults.forEach(item => item.score /= sum);

    const response = {
      sequence: text,
      labels: normalizedResults.map(r => r.category),
      scores: normalizedResults.map(r => r.score),
      filteredOut: false
    };
    
    if (returnFilterInfo && filterResult) {
      response.filterResult = filterResult;
    }
    
    return response;
  }
}
