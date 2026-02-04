import { NLIClassificationPipeline, labels } from './NLI.js';
import { FewShotClassifier } from './FeatureEmbedded.js';

// Choose which classifier to test: 'nli' or 'few-shot'
const CLASSIFIER_TYPE = 'nli';
// const CLASSIFIER_TYPE = 'few-shot';

// Define label groups - related labels that should be considered acceptable matches
const labelGroups = {
  'system_monitoring': [
    'The user is asking about system configuration',
    'The user is asking about infrastructure monitoring'
  ],
  'class_analysis': [
    'The user wants analysis about a class performance',
    'The user wants statistics about a class performance'
  ],
  'course_analysis': [
    'The user wants analysis about a course performance',
    'The user wants statistics about a course performance'
  ],
  'quiz_creation': [
    'The user wants to create quizzes',
    'The user wants to design quizzes'
  ],
  'assessment_creation': [
    'The user wants to create assessments',
    'The user wants to design assessments'
  ],
  'learning_help': [
    'The user wants explanations about learning content',
    'The user wants tutoring about learning content'
  ],
  'personalized_learning': [
    'The user wants personalized learning recommendations',
    'The user wants personalized practice'
  ]
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
    expected: "The user wants explanations about learning content"
  },
  {
    input: "analyze the scores from class L01",
    expected: "The user wants analysis about a class performance"
  },
  {
    input: "check if the server is running properly",
    expected: "The user is asking about infrastructure monitoring"
  },
  {
    input: "create a quiz about Python programming",
    expected: "The user wants to create quizzes"
  },
  {
    input: "what topics should I study next based on my progress",
    expected: "The user wants personalized learning recommendations"
  },
  {
    input: "what is the weather like today",
    expected: "The user is talking about something unrelated to education"
  },
  {
    input: "help me understand calculus derivatives",
    expected: "The user wants explanations about learning content"
  },
  {
    input: "generate assessment questions for biology chapter 3",
    expected: "The user wants to create assessments"
  },
  {
    input: "show statistics about student performance in math",
    expected: "The user wants statistics about a course performance"
  },
  {
    input: "monitor the system uptime",
    expected: "The user is asking about infrastructure monitoring"
  }
];

async function runTests() {
  console.log('═══════════════════════════════════════════════════════════════');
  console.log(`TESTING ${CLASSIFIER_TYPE.toUpperCase()} CLASSIFIER`);
  console.log('═══════════════════════════════════════════════════════════════\n');

  // Load the classifier
  let classifier;
  if (CLASSIFIER_TYPE === 'nli') {
    classifier = await NLIClassificationPipeline.getInstance();
  } else {
    await FewShotClassifier.getInstance();
  }

  let totalTests = testCases.length;
  let correctFirst = 0;
  let correctSecond = 0;

  for (let i = 0; i < testCases.length; i++) {
    const testCase = testCases[i];
    
    console.log('─────────────────────────────────────────────────────────────');
    console.log(`\n[Test ${i + 1}/${totalTests}]`);
    console.log(`Input: "${testCase.input}"`);
    console.log(`Expected: ${testCase.expected}`);

    // Get classification results
    let result;
    if (CLASSIFIER_TYPE === 'nli') {
      result = await classifier(testCase.input, labels);
    } else {
      result = await FewShotClassifier.classify(testCase.input);
    }

    // Get top 2 predictions
    const first = {
      label: result.labels[0],
      score: result.scores[0]
    };
    const second = {
      label: result.labels[1],
      score: result.scores[1]
    };

    // Display results
    const firstPercentage = (first.score * 100).toFixed(2);
    const secondPercentage = (second.score * 100).toFixed(2);

    console.log(`1st: ${first.label}: ${firstPercentage}%`);
    console.log(`2nd: ${second.label}: ${secondPercentage}%`);

    // Check accuracy (exact match or related label)
    const isFirstExact = first.label === testCase.expected;
    const isFirstRelated = areLabelsRelated(first.label, testCase.expected);
    const isSecondExact = second.label === testCase.expected;
    const isSecondRelated = areLabelsRelated(second.label, testCase.expected);

    if (isFirstExact) {
      correctFirst++;
      console.log('✓ CORRECT (1st match - exact)');
    } else if (isFirstRelated) {
      correctFirst++;
      console.log('✓ CORRECT (1st match - related)');
    } else if (isSecondExact) {
      correctSecond++;
      console.log('⚠ PARTIAL (2nd match - exact)');
    } else if (isSecondRelated) {
      correctSecond++;
      console.log('⚠ PARTIAL (2nd match - related)');
    } else {
      console.log('✗ INCORRECT');
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
  console.log('═══════════════════════════════════════════════════════════════\n');
}

// Run the tests
runTests().catch(console.error);
