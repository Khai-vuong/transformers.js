import { pipeline, env } from '@xenova/transformers';

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
  
  // Routing threshold - must reach 50% similarity to route to a module
  static routingThreshold = 0.5;
  
  // Role-based heuristic coefficients (boost percentages)
  static roleCoefficients = {
    'admin': {
      'system_configuration': 0.10,
      'data_analysis': 0.03,
      'quiz_creation': 0.00
    },
    'lecturer': {
      'system_configuration': 0.00,
      'data_analysis': 0.05,
      'quiz_creation': 0.10
    },
    'student': {
      'system_configuration': 0.00,
      'data_analysis': 0.00,
      'quiz_creation': 0.00
    }
  };

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
  }

  static cosineSimilarity(vecA, vecB) {
    let dotProduct = 0;
    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
    }
    return dotProduct; // Vectors are already normalized
  }


  static applyRoleHeuristic(categoryScores, role) {
    if (!role) return categoryScores;
    
    const normalizedRole = role.toLowerCase();
    const coefficients = this.roleCoefficients[normalizedRole];
    
    if (!coefficients) {
      console.warn(`Unknown role: ${role}. No heuristic applied.`);
      return categoryScores;
    }
    
    const boostedScores = {};
    for (const [category, score] of Object.entries(categoryScores)) {
      const boost = coefficients[category] || 0;
      boostedScores[category] = score + boost;
    }
    
    return boostedScores;
  }

  static async classify(text, role = null) {
    await this.getInstance();

    // Get embedding for the input text
    const textEmbedding = await this.embeddingInstance(text, { pooling: 'mean', normalize: true });
    const textVector = textEmbedding.data;

    // Calculate similarity with each category centroid
    const categoryScores = {};
    
    for (const [category, centroidEmbedding] of Object.entries(this.categoryEmbeddings)) {
      categoryScores[category] = this.cosineSimilarity(textVector, centroidEmbedding);
    }

    // Apply role-based heuristic boost
    const boostedScores = this.applyRoleHeuristic(categoryScores, role);

    // Sort categories by boosted score
    const sortedCategories = Object.entries(boostedScores)
      .sort((a, b) => b[1] - a[1])
      .map(([category, score]) => ({ category, score }));

    // Check if the best match meets the threshold
    let bestMatch;

    if (sortedCategories[0].score < this.routingThreshold) {
        bestMatch = { category: 'outer api', score: 1.0 - sortedCategories[0].score };
    } else {
        bestMatch = sortedCategories[0];
    }


    let decision;
    
    if (bestMatch.score >= this.routingThreshold) {
      decision = `forward to ${bestMatch.category} module`;
    } else {
      decision = 'forward to outer API';
    }

    return {
      sequence: text,
      decision: decision,
      bestMatch: {
        category: bestMatch.category,
        score: bestMatch.score
      },
      allScores: sortedCategories,
      threshold: this.routingThreshold,
      role: role,
      originalScores: Object.entries(categoryScores)
        .sort((a, b) => b[1] - a[1])
        .map(([category, score]) => ({ category, score }))
    };
  }
}
