import { pipeline, env } from '@xenova/transformers';

// Binary filter labels for identifying e-learning related vs unrelated messages
export const labels = [
  'education and learning',
  'unrelated to education'
];


export class NLIClassificationPipeline {
  static task = 'zero-shot-classification';
//   static model = 'Xenova/bart-large-mnli';
  // Alternative NLI models:
  // static model = 'Xenova/distilbert-base-uncased-mnli';
  static model = 'Xenova/mobilebert-uncased-mnli';
  static instance = null;
  static hypothesisTemplate = 'The user is talking about {}.';
  
  // Threshold for determining if message is related (confidence score)
  static relatedThreshold = 0.5;

  static async getInstance(progress_callback = null) {
    if (this.instance === null) {
      // NOTE: Uncomment this to change the cache directory
      // env.cacheDir = './.cache';

      this.instance = pipeline(this.task, this.model, { progress_callback });
    }

    return this.instance;
  }

  static async classify(text, candidateLabels, hypothesisTemplate = null) {
    const classifier = await this.getInstance();
    const template = hypothesisTemplate || this.hypothesisTemplate;
    
    return await classifier(text, candidateLabels, { 
      hypothesis_template: template 
    });
  }

  /**
   * Filter method to determine if a message is related to e-learning system.
   * Returns an object with isRelated boolean and confidence scores.
   * 
   * @param {string} text - The input text to classify
   * @returns {Promise<{isRelated: boolean, confidence: number, category: string, scores: object}>}
   */
  static async isRelatedToELearning(text) {
    const result = await this.classify(text, labels);
    
    // Find the index of each category
    const relatedIndex = result.labels.indexOf('education and learning');
    const unrelatedIndex = result.labels.indexOf('unrelated to education');
    
    const relatedScore = result.scores[relatedIndex];
    const unrelatedScore = result.scores[unrelatedIndex];
    
    // Determine if message is related based on threshold
    const isRelated = relatedScore >= this.relatedThreshold;
    
    return {
      isRelated: isRelated,
      confidence: isRelated ? relatedScore : unrelatedScore,
      category: isRelated ? 'education and learning' : 'unrelated to education',
      scores: {
        'education and learning': relatedScore,
        'unrelated to education': unrelatedScore
      },
      rawResult: result
    };
  }

  
}
