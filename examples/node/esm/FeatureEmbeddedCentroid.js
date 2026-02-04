import { pipeline, env } from '@xenova/transformers';

// Few-shot examples: provide example texts for each category
export const fewShotExamples = {
  'system_configuration': [
    'Check the server status',
    'Is the system running properly?',
    'Monitor the infrastructure health',
    'What is the uptime of the service?'
  ],
  'data_analysis': [
    'Analyze the scores from class L01',
    'Show me statistics about student performance',
    'What are the average grades?',
    'Generate a report on quiz results'
  ],
  'quiz_creation': [
    'Create a quiz about mathematics',
    'Generate assessment questions for chapter 5',
    'Design a test for students',
    'Make practice questions about photosynthesis'
  ],
  'content_tutoring': [
    'Explain the Pythagorean theorem',
    'Explain the concept of gravity',
    'How does photosynthesis work?',
    'Help me understand calculus',
    'What is the meaning of oxidation?',
    'Why does imaginary numbers exist?'
  ],
  'personalized_learning': [
    'Recommend study materials for me',
    'What should I practice next?',
    'Give me personalized exercises',
    'Suggest advance topics based on my progress',
  ],
  'unrelated': [
    'What is the weather today?',
    'Tell me a joke',
    'What time is it?',
    'How to cook pasta?',
    
  ]
};

export class FeatureEmbeddedClassifier {
  static embeddingModel = 'Xenova/all-MiniLM-L6-v2';
  static embeddingInstance = null;
  static categoryEmbeddings = null;
  static systemCentroid = null; // Centroid of all education-related centroids
  
  // Filter options
  static useSystemCentroidFilter = true; // Enable/disable centroid-based pre-filtering
  static centroidSimilarityThreshold = 0.3; // Minimum similarity to system centroid (50%)

  static async getInstance(progress_callback = null) {
    if (this.embeddingInstance === null) {
      // NOTE: Uncomment this to change the cache directory
      // env.cacheDir = './.cache';

      this.embeddingInstance = await pipeline('feature-extraction', this.embeddingModel, { progress_callback });
      
      // Pre-compute centroid embeddings for all examples
      await this.computeCategoryEmbeddings();
    }

    return this.embeddingInstance;
  }

  static async computeCategoryEmbeddings() {
    if (this.categoryEmbeddings !== null) return;

    this.categoryEmbeddings = {};

    // Compute centroid (average) embedding for each category
    // Visualization for categoryEmbeddings structure:
    // {
    //   'system_configuration': [centroid_embedding],
    //   'data_analysis': [centroid_embedding],
    //   ...
    // }
    for (const [category, examples] of Object.entries(fewShotExamples)) {
      // Get embeddings for all examples in this category
      const embeddings = [];
      for (const example of examples) {
        const result = await this.embeddingInstance(example, { pooling: 'mean', normalize: true });
        embeddings.push(result.data);
      }
      
      // Compute centroid: average all embeddings element-wise
      const embeddingDim = embeddings[0].length;
      const centroid = new Array(embeddingDim).fill(0);
      
      for (const embedding of embeddings) {
        for (let i = 0; i < embeddingDim; i++) {
          centroid[i] += embedding[i];
        }
      }
      
      // Average and normalize the centroid
      for (let i = 0; i < embeddingDim; i++) {
        centroid[i] /= embeddings.length;
      }
      
      // Normalize centroid to unit vector
      const magnitude = Math.sqrt(centroid.reduce((sum, val) => sum + val * val, 0));
      for (let i = 0; i < embeddingDim; i++) {
        centroid[i] /= magnitude;
      }
      
      // Store single centroid embedding for this category
      this.categoryEmbeddings[category] = centroid;
    }
    
    // Compute system centroid: centroid of all education-related centroids (excluding 'unrelated')
    this.computeSystemCentroid();
  }

  static computeSystemCentroid() {
    // Get all education-related centroids (exclude 'unrelated')
    const educationCentroids = [];
    for (const [category, centroid] of Object.entries(this.categoryEmbeddings)) {
      if (category !== 'unrelated') {
        educationCentroids.push(centroid);
      }
    }
    
    if (educationCentroids.length === 0) {
      console.warn('No education-related centroids found for system centroid computation');
      return;
    }
    
    // Compute centroid of centroids (super-centroid representing the entire education system)
    const embeddingDim = educationCentroids[0].length;
    const systemCentroid = new Array(embeddingDim).fill(0);
    
    for (const centroid of educationCentroids) {
      for (let i = 0; i < embeddingDim; i++) {
        systemCentroid[i] += centroid[i];
      }
    }
    
    // Average
    for (let i = 0; i < embeddingDim; i++) {
      systemCentroid[i] /= educationCentroids.length;
    }
    
    // Normalize to unit vector
    const magnitude = Math.sqrt(systemCentroid.reduce((sum, val) => sum + val * val, 0));
    for (let i = 0; i < embeddingDim; i++) {
      systemCentroid[i] /= magnitude;
    }
    
    this.systemCentroid = systemCentroid;
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
      skipFilter = false, // Set to true to bypass system centroid filter
      returnFilterInfo = true // Set to true to include filter information in response
    } = options;
    
    await this.getInstance();

    // Get embedding for the input text
    const textEmbedding = await this.embeddingInstance(text, { pooling: 'mean', normalize: true });
    const textVector = textEmbedding.data;

    // Step 1: System Centroid Filter (optional, but recommended)
    let filterResult = null;
    if (this.useSystemCentroidFilter && !skipFilter && this.systemCentroid) {
      // Calculate similarity to the system centroid (represents all education-related content)
      const systemSimilarity = this.cosineSimilarity(textVector, this.systemCentroid);
      
      filterResult = {
        systemSimilarity: systemSimilarity,
        threshold: this.centroidSimilarityThreshold,
        isRelated: systemSimilarity >= this.centroidSimilarityThreshold
      };
      
      // If message is unrelated to education system, return early
      if (!filterResult.isRelated) {
        const response = {
          sequence: text,
          labels: ['unrelated'],
          scores: [1.0],
          filteredOut: true,
          reason: 'Message not related to e-learning system (low similarity to system centroid)'
        };
        
        if (returnFilterInfo) {
          response.filterResult = filterResult;
        }
        
        return response;
      }
    }

    // Step 2: Category Centroid Classification (only for education-related messages)
    // Calculate similarity with each category centroid
    const categoryScores = {};
    
    for (const [category, centroidEmbedding] of Object.entries(this.categoryEmbeddings)) {
      // Skip 'unrelated' category if filter is enabled (we already filtered it out)
      if (this.useSystemCentroidFilter && category === 'unrelated') {
        continue;
      }
      
      // Compare directly to the category centroid (single comparison per category)
      categoryScores[category] = this.cosineSimilarity(textVector, centroidEmbedding);
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
