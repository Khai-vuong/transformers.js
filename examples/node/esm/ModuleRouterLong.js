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
  static summarizationModel = 'Xenova/distilbart-cnn-6-6';
  static embeddingInstance = null;
  static summarizationInstance = null;
  static categoryEmbeddings = null;
  
  // Long text processing thresholds
  static longTextThreshold = 15; // words
  static minSentenceLength = 5; // words
  
  // Discourse chunking thresholds
  static chunkSimilarityThreshold = 0.65; // cosine similarity threshold for grouping sentences into chunks
  static chunkSignificanceThreshold = 0.25; // minimum score to count as "significant appearance" in coverage
  static decisionThreshold = 0.30; // minimum score for an intent to be included in results
  
  // Position weighting - how much to boost later chunks (e.g., 0.5 = 50% boost for last chunk)
  static positionBoost = 0.5;
  
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

  static async getSummarizationInstance(progress_callback = null) {
    if (this.summarizationInstance === null) {
      this.summarizationInstance = await pipeline('summarization', this.summarizationModel, { progress_callback });
    }
    return this.summarizationInstance;
  }

  static splitIntoSentences(text) {
    // Split by sentence delimiters (., !, ?)
    const sentences = text
      .split(/[.!?]+/)
      .map(s => s.trim())
      .filter(s => {
        const wordCount = s.split(/\s+/).length;
        return wordCount >= this.minSentenceLength;
      });
    
    return sentences;
  }

  static countWords(text) {
    const trimmed = text.trim();
    if (!trimmed) return 0;
    return trimmed.split(/\s+/).length;
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

  static async classifySingleText(text) {
    // Get embedding for the input text
    const textEmbedding = await this.embeddingInstance(text, { pooling: 'mean', normalize: true });
    const textVector = textEmbedding.data;

    // Calculate similarity with each category centroid
    const categoryScores = {};
    
    for (const [category, centroidEmbedding] of Object.entries(this.categoryEmbeddings)) {
      categoryScores[category] = this.cosineSimilarity(textVector, centroidEmbedding);
    }

    return categoryScores;
  }

  static async classifyLongText(text, role = null) {
    console.log('\nLong text detected - using discourse chunking classification...\n');

    // Step 1: Split into sentences
    const sentences = this.splitIntoSentences(text);
    console.log(`Split into ${sentences.length} sentences`);
    console.log('------------------------------------------------------------');
    sentences.forEach((s, i) => console.log(`  [${i}] ${s}`));
    
    if (sentences.length === 0) {
      return await this.classifySingleText(text);
    }

    // Step 2: Compute sentence embeddings and create discourse chunks
    console.log('\n--- Step 2: Discourse Chunking ---');
    const sentenceEmbeddings = [];
    for (const sentence of sentences) {
      const embedding = await this.embeddingInstance(sentence, { pooling: 'mean', normalize: true });
      sentenceEmbeddings.push(embedding.data);
    }

    // Create chunks based on cosine similarity between consecutive sentences
    const chunks = [];
    let currentChunk = [0]; // Start with first sentence
    
    for (let i = 1; i < sentences.length; i++) {
      const similarity = this.cosineSimilarity(sentenceEmbeddings[i-1], sentenceEmbeddings[i]);
      console.log(`Similarity [${i-1}]-[${i}]: ${(similarity * 100).toFixed(1)}%`);
      
      if (similarity >= this.chunkSimilarityThreshold) {
        // Add to current chunk
        currentChunk.push(i);
      } else {
        // Start new chunk
        chunks.push(currentChunk);
        currentChunk = [i];
      }
    }
    // Don't forget the last chunk
    chunks.push(currentChunk);

    console.log(`\nCreated ${chunks.length} discourse chunks:`);
    chunks.forEach((chunk, idx) => {
      const chunkText = chunk.map(sentIdx => sentences[sentIdx]).join(' ');
      console.log(`  Chunk ${idx + 1}: [sentences ${chunk.join(', ')}]`);
      console.log(`    Text: "${chunkText}"`);
    });

    // Step 3: Classify each chunk with category centroids
    console.log('\n--- Step 3: Classify Each Chunk ---');
    const chunkClassifications = [];
    
    for (let chunkIdx = 0; chunkIdx < chunks.length; chunkIdx++) {
      const chunk = chunks[chunkIdx];
      const chunkText = chunk.map(sentIdx => sentences[sentIdx]).join(' ');
      
      // Classify this chunk
      const scores = await this.classifySingleText(chunkText);
      
      // Apply role heuristic
      const boostedScores = this.applyRoleHeuristic(scores, role);
      
      // Find best category for this chunk
      const sortedScores = Object.entries(boostedScores)
        .sort((a, b) => b[1] - a[1]);
      
      console.log(`\n  Chunk ${chunkIdx + 1}:`);
      console.log(`    Top 3: ${sortedScores.slice(0, 3).map(([cat, score]) => 
        `${cat}(${(score * 100).toFixed(1)}%)`).join(', ')}`);
      
      chunkClassifications.push({
        chunkIndex: chunkIdx,
        text: chunkText,
        scores: boostedScores,
        position: chunks.length > 1 ? chunkIdx / (chunks.length - 1) : 0 // 0 to 1
      });
    }

    // Step 4: Rank using multiple signals
    console.log('\n--- Step 4: Aggregate with Ranking Signals ---');
    
    const intentStats = {};
    const categories = Object.keys(this.categoryEmbeddings);
    
    // Initialize stats for each category
    for (const category of categories) {
      intentStats[category] = {
        maxConfidence: 0,
        chunkAppearances: [],
        totalScore: 0,
        positionWeightedScore: 0
      };
    }
    
    // Collect statistics from each chunk
    for (const classification of chunkClassifications) {
      const positionWeight = 1 + (classification.position * this.positionBoost);
      
      for (const category of categories) {
        const score = classification.scores[category];
        const stats = intentStats[category];
        
        // Max confidence signal
        if (score > stats.maxConfidence) {
          stats.maxConfidence = score;
        }
        
        // Coverage signal - track which chunks this intent appears in
        stats.chunkAppearances.push({
          chunkIndex: classification.chunkIndex,
          score: score
        });
        
        stats.totalScore += score;
        
        // Position weighting - later chunks get boosted (call-to-action)
        stats.positionWeightedScore += score * positionWeight;
      }
    }
    
    // Calculate final scores
    const finalScores = {};
    const numChunks = chunks.length;
    
    console.log('\nSignal Analysis:');
    for (const category of categories) {
      const stats = intentStats[category];
      
      // Coverage: how many chunks have this intent with meaningful score
      const significantAppearances = stats.chunkAppearances.filter(
        app => app.score >= this.chunkSignificanceThreshold
      ).length;
      const coverageRatio = significantAppearances / numChunks;
      
      // Combine signals:
      // - Max confidence (40% weight) - strongest signal
      // - Position weighted average (40% weight) - considers call-to-action
      // - Coverage boost (20% weight) - recurring themes
      const avgPositionWeighted = stats.positionWeightedScore / numChunks;
      
      finalScores[category] = 
        (stats.maxConfidence * 0.4) +
        (avgPositionWeighted * 0.4) +
        (coverageRatio * 0.2);
      
      console.log(`  ${category}:`);
      console.log(`    Max Confidence: ${(stats.maxConfidence * 100).toFixed(1)}%`);
      console.log(`    Coverage: ${significantAppearances}/${numChunks} chunks (${(coverageRatio * 100).toFixed(1)}%)`);
      console.log(`    Position-Weighted Avg: ${(avgPositionWeighted * 100).toFixed(1)}%`);
      console.log(`    Final Score: ${(finalScores[category] * 100).toFixed(1)}%`);
    }
    
    // Step 5: Filter by decision threshold and return sorted list
    console.log(`\n--- Step 5: Filter & Sort (Threshold: ${(this.decisionThreshold * 100)}%) ---`);
    
    const qualifyingIntents = Object.entries(finalScores)
      .filter(([category, score]) => score >= this.decisionThreshold)
      .sort((a, b) => b[1] - a[1])
      .map(([category, score]) => ({
        category,
        score,
        details: {
          maxConfidence: intentStats[category].maxConfidence,
          coverage: intentStats[category].chunkAppearances.filter(
            app => app.score >= this.chunkSignificanceThreshold
          ).length,
          totalChunks: numChunks
        }
      }));
    
    console.log(`\n${qualifyingIntents.length} intent(s) passed the threshold:`);
    qualifyingIntents.forEach((intent, idx) => {
      console.log(`  ${idx + 1}. ${intent.category}: ${(intent.score * 100).toFixed(1)}% ` +
        `[max: ${(intent.details.maxConfidence * 100).toFixed(1)}%, ` +
        `coverage: ${intent.details.coverage}/${intent.details.totalChunks}]`);
    });
    
    // Return result in format compatible with existing code
    // Convert to simple scores object for backward compatibility
    const scoresObject = {};
    for (const category of categories) {
      scoresObject[category] = finalScores[category];
    }
    
    return scoresObject;
  }

  static async classify(text, role = null) {
    await this.getInstance();

    const wordCount = this.countWords(text);
    let categoryScores;

    console.log(`\nClassifying text with ${wordCount} words...`);

    // Decide whether to use long-text processing
    if (wordCount > this.longTextThreshold) {
        console.log('Text exceeds long text threshold, using long text classification.');
        categoryScores = await this.classifyLongText(text, role);
    } else {
        console.log('Text within normal length, using single text classification.');
        categoryScores = await this.classifySingleText(text);
    }

    // Apply role-based heuristic boost
    const boostedScores = this.applyRoleHeuristic(categoryScores, role);

    // Sort categories by boosted score
    const sortedCategories = Object.entries(boostedScores)
      .sort((a, b) => b[1] - a[1])
      .map(([category, score]) => ({ category, score }));

    // Check if the best match meets the threshold
    const topCategory = sortedCategories[0];
    let decision;
    let bestMatch;

    if (topCategory.score >= this.routingThreshold) {
      // Route to the module
      decision = `forward to ${topCategory.category} module`;
      bestMatch = topCategory;
    } else {
      // Route to outer API
      decision = 'forward to outer API';
      bestMatch = { category: 'outer api', score: 1.0 - topCategory.score };
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
