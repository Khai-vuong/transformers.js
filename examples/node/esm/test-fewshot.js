// import { FeatureEmbeddedClassifier, fewShotExamples } from './FeatureEmbedded.js';
import { FeatureEmbeddedClassifier, fewShotExamples } from './FeatureEmbeddedCentroid.js';

// Define label groups for few-shot categories
const labelGroups = {
  'system_operations': ['system_configuration'],
  'data_operations': ['data_analysis'],
  'assessment_creation': ['quiz_creation'],
  'learning_support': ['content_tutoring', 'personalized_learning'],
  'off_topic': ['unrelated']
};

// Helper function to check if two labels are in the same group
function areLabelsRelated(label1, label2) {
  for (const group of Object.values(labelGroups)) {
    if (group.includes(label1) && group.includes(label2)) {
      return true;
    }
  }
  return false;
}

// Define test cases with input text and expected category
const testCases = [
  {
    input: "explain the concept of computer network",
    expected: "content_tutoring"
  },
  {
    input: "analyze the scores from class L01",
    expected: "data_analysis"
  },
  {
    input: "check if the server is running properly",
    expected: "system_configuration"
  },
  {
    input: "create a quiz about Python programming",
    expected: "quiz_creation"
  },
  {
    input: "what topics should I study next based on my progress",
    expected: "personalized_learning"
  },
  {
    input: "what is the weather like today",
    expected: "unrelated"
  },
  {
    input: "help me understand calculus derivatives",
    expected: "content_tutoring"
  },
  {
    input: "generate assessment questions for biology chapter 3",
    expected: "quiz_creation"
  },
  {
    input: "show statistics about student performance in math",
    expected: "data_analysis"
  },
  {
    input: "monitor the system uptime",
    expected: "system_configuration"
  },
  {
    input: "What is the difference between mitosis and meiosis?",
    expected: "content_tutoring"
  },
  {
    input: "Design a multiple choice test for history",
    expected: "quiz_creation"
  },
  {
    input: "I need practice problems for algebra",
    expected: "personalized_learning"
  },
  {
    input: "Calculate the average test score for the semester",
    expected: "data_analysis"
  },
  {
    input: "Is the learning management system online?",
    expected: "system_configuration"
  },
  {
    input: "How do I make a sandwich?",
    expected: "unrelated"
  },
  {
    input: "Recommend courses based on my interests",
    expected: "personalized_learning"
  },
  {
    input: "Show me the grade distribution for this class",
    expected: "data_analysis"
  }
];

async function runTests() {
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('TESTING FEW-SHOT CLASSIFIER');
  console.log('═══════════════════════════════════════════════════════════════');
  
  // Display categories and their examples
  console.log('\n📚 Few-Shot Categories and Examples:');
  console.log('─────────────────────────────────────────────────────────────');
  for (const [category, examples] of Object.entries(fewShotExamples)) {
    console.log(`\n${category}:`);
    examples.forEach(ex => console.log(`  • ${ex}`));
  }
  console.log('\n═══════════════════════════════════════════════════════════════\n');

  // Load the classifier
  console.log('⏳ Loading few-shot classifier and computing embeddings...\n');
  await FeatureEmbeddedClassifier.getInstance();
  console.log('✓ Classifier ready!\n');

  let totalTests = testCases.length;
  let correctFirst = 0;
  let correctSecond = 0;
  let strongMatches = 0;

  for (let i = 0; i < testCases.length; i++) {
    const testCase = testCases[i];
    
    console.log('─────────────────────────────────────────────────────────────');
    console.log(`\n[Test ${i + 1}/${totalTests}]`);
    console.log(`Input: "${testCase.input}"`);
    console.log(`Expected: ${testCase.expected}`);

    // Get classification results
    const result = await FeatureEmbeddedClassifier.classify(testCase.input);

    // Get top 3 predictions (few-shot might have less confident predictions)
    const first = {
      label: result.labels[0],
      score: result.scores[0]
    };
    const second = {
      label: result.labels[1],
      score: result.scores[1]
    };
    const third = {
      label: result.labels[2],
      score: result.scores[2]
    };

    // Display results
    const firstPercentage = (first.score * 100).toFixed(2);
    const secondPercentage = (second.score * 100).toFixed(2);
    const thirdPercentage = (third.score * 100).toFixed(2);

    console.log(`\nPredictions:`);
    console.log(`  1st: ${first.label.padEnd(25)} ${firstPercentage}%`);
    console.log(`  2nd: ${second.label.padEnd(25)} ${secondPercentage}%`);
    console.log(`  3rd: ${third.label.padEnd(25)} ${thirdPercentage}%`);

    if (first.score >= 0.75) {
        console.log(' :3 (Really strong match for 1st place!)');
        strongMatches++;
    }
    

    // Check accuracy (exact match or related label)
    const isFirstExact = first.label === testCase.expected;
    const isFirstRelated = areLabelsRelated(first.label, testCase.expected);
    const isSecondExact = second.label === testCase.expected;
    const isSecondRelated = areLabelsRelated(second.label, testCase.expected);

    if (isFirstExact) {
      correctFirst++;
      console.log('\n✓ CORRECT (1st match - exact)');
    } else if (isFirstRelated) {
      correctFirst++;
      console.log('\n✓ CORRECT (1st match - related)');
    } else if (isSecondExact) {
      correctSecond++;
      console.log('\n⚠ PARTIAL (2nd match - exact)');
    } else if (isSecondRelated) {
      correctSecond++;
      console.log('\n⚠ PARTIAL (2nd match - related)');
    } else {
      console.log('\n✗ INCORRECT');
    }
  }

  // Display summary
  console.log('\n═══════════════════════════════════════════════════════════════');
  console.log('SUMMARY');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log(`Total Tests: ${totalTests}`);
  console.log(`1st Position Accuracy: ${correctFirst}/${totalTests} (${(correctFirst/totalTests*100).toFixed(2)}%)`);
  console.log(`2nd Position Matches: ${correctSecond}/${totalTests} (${(correctSecond/totalTests*100).toFixed(2)}%)`);
  console.log(`Combined (Top-2) Accuracy: ${correctFirst + correctSecond}/${totalTests} (${((correctFirst + correctSecond)/totalTests*100).toFixed(2)}%)`);
  console.log(`Failed: ${totalTests - correctFirst - correctSecond}/${totalTests}`);
  console.log(`Strong Matches (1st place ≥ 75% confidence): ${strongMatches}/${totalTests}`);
  console.log('═══════════════════════════════════════════════════════════════\n');

  // Additional insights
//   console.log('💡 Few-Shot Learning Insights:');
//   console.log('─────────────────────────────────────────────────────────────');
//   console.log('Few-shot classification uses semantic similarity between');
//   console.log('input text and example embeddings for each category.');
//   console.log('To improve accuracy:');
//   console.log('  • Add more diverse examples to each category');
//   console.log('  • Ensure examples clearly represent each category');
//   console.log('  • Use examples similar to expected real-world inputs');
//   console.log('═══════════════════════════════════════════════════════════════\n');
}

// Run the tests
runTests().catch(console.error);
