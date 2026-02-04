import { FeatureEmbeddedClassifier } from './ModuleRouterLong.js';

// Test cases with long paragraphs
const longTextTests = [
  {
    input: `Our e-learning platform has been experiencing some issues lately. 
    The server seems to be running slow during peak hours, especially between 2-4 PM. 
    We need to check the system health and monitor the uptime. 
    Additionally, I've noticed that the database connection drops occasionally. 
    Can you investigate the infrastructure and make sure everything is properly configured? 
    We should also look at the CPU and memory usage to identify any bottlenecks.`,
    expected: "system_configuration",
    description: "Long paragraph about system issues",
    role: "Admin"
  },
  {
    input: `I need help creating educational content for my students. 
    First, I want to design a comprehensive quiz about linear algebra, covering topics like matrices, vectors, and eigenvalues. 
    Then, I need to generate practice problems for calculus, focusing on derivatives and integrals. 
    After that, could you help me create test questions for biology chapter 3, specifically about cell division and genetics? 
    Finally, I'd like to make a history test covering World War II and its impact on global politics.`,
    expected: "quiz_creation",
    description: "Long paragraph about creating multiple quizzes",
    role: "Lecturer"
  },
  {
    input: `The end of semester is approaching and I need to analyze student performance across all my classes. 
    Can you show me the statistics for class L01, including average grades and score distribution? 
    I also want to generate a comprehensive report comparing performance between L01 and L02. 
    Additionally, I'd like to see the grade trends over the past three months. 
    It would be helpful to calculate the median scores for each assignment and identify students who are struggling. 
    Finally, can you create an analytics dashboard showing the overall progress of all students in my courses?`,
    expected: "data_analysis",
    description: "Long paragraph about analyzing student data",
    role: "Lecturer"
  },
  {
    input: `I'm planning a vacation to Europe next summer. 
    I want to visit Paris, Rome, and Barcelona, spending about a week in each city. 
    Can you recommend good hotels that are centrally located? 
    Also, what's the best time to book flights to get the cheapest prices? 
    I'm interested in trying authentic local cuisine, so please suggest some restaurants. 
    What are the must-see attractions in each city? 
    Should I rent a car or use public transportation?`,
    expected: "outer_api",
    description: "Long paragraph about vacation planning (unrelated)",
    role: "Student"
  },
  {
    input: `I'm a new student and I have several questions about my studies. 
    What are the best study techniques for memorizing complex information? 
    How can I improve my time management skills to balance coursework and extracurricular activities? 
    Should I focus on computer science or business administration for my major? 
    How long does it typically take to learn Python programming? 
    Also, what books would you recommend for improving my vocabulary? 
    I'm finding it hard to stay motivated during exam season.`,
    expected: "gray_zone",
    description: "Long paragraph with mixed general education questions",
    role: "Student"
  },
  {
    input: `We need to conduct a comprehensive system audit. 
    First, check if all servers are running properly and monitor their health status. 
    Then, analyze the student performance data from last semester and generate detailed reports. 
    After that, create practice quizzes for the upcoming midterm exams. 
    Finally, review the uptime logs and identify any system bottlenecks. 
    This is urgent as we're approaching the end of the academic year.`,
    expected: "mixed",
    description: "Long paragraph with multiple intents (system, analysis, quiz)",
    role: "Admin"
  },
  {
    input: `Hi team, the LMS has been slow since Monday and students are reporting timeouts when they submit assignments. 
    Can you check the server health, review the error logs, and verify the database connection pool settings? 
    If CPU or memory is spiking, please capture metrics and suggest whether we should scale up during peak hours. 
    Also, confirm that the nightly backup jobs are still completing successfully.`,
    expected: "system_configuration",
    description: "Long paragraph about monitoring and fixing LMS infrastructure",
    role: "Admin"
  },
  {
    input: `I'm planning next week's review session and want a structured set of materials. 
    Please create a 20-question quiz on probability basics, then add a short section of practice problems about conditional probability and Bayes' theorem. 
    After that, build a separate mini-test for statistics covering mean, variance, and standard deviation, and include an answer key for each part.`,
    expected: "quiz_creation",
    description: "Long paragraph focused on generating quizzes and practice problems",
    role: "Lecturer"
  },
  {
    input: `Could you analyze how my students performed in the last two assignments and summarize the trend? 
    I want the average, median, and a quick distribution by score band, plus a comparison between sections A and B. 
    If possible, flag any outliers and list students who dropped more than 10 points from the previous quiz.`,
    expected: "data_analysis",
    description: "Long paragraph about analytics and performance trends",
    role: "Lecturer"
  },
  {
    input: `I'm trying to prepare for finals and I'm not sure how to organize my study plan. 
    I need advice on how to break down a long chapter into daily tasks, how to stay focused, and which study techniques work best for problem-solving. 
    Also, can you recommend some resources for improving academic writing and vocabulary?`,
    expected: "gray_zone",
    description: "Long paragraph with general study advice requests",
    role: "Student"
  },
  {
    input: `Next week I have to present a report to the department, so I need you to summarize overall student performance this term. 
    Please calculate the pass rate, show the grade distribution for each class, and highlight any classes with unusually low averages. 
    After that, create a one-page narrative summary that I can paste into the report.`,
    expected: "data_analysis",
    description: "Long paragraph about compiling a performance report",
    role: "Lecturer"
  },
  {
    input: `I want to design a new assessment pack for our biology unit. 
    Create a quiz on cell division and genetics, then add a second section with short-answer questions about DNA replication. 
    Finally, give me a few higher-order questions that assess critical thinking rather than memorization.`,
    expected: "quiz_creation",
    description: "Long paragraph about building multi-part biology assessments",
    role: "Lecturer"
  },
  {
    input: `We need to triage an issue and also plan content for next month. 
    First, check the uptime dashboard and verify that the API gateway isn't throttling legitimate traffic. 
    Then, analyze last week's quiz results to see which topics students struggled with, and use that to create a new practice quiz on those weak areas. 
    If you have time, summarize the top three pain points from support tickets.`,
    expected: "mixed",
    description: "Long paragraph mixing system checks, analytics, and quiz creation",
    role: "Admin"
  },
  {
    input: `I'm organizing a summer trip to Japan with a couple of friends. 
    We want to split time between Tokyo, Kyoto, and Osaka, and we care about food tours and day trips to nearby towns. 
    Can you suggest a good itinerary, where to stay for easy transit, and whether a rail pass is worth it for a 10-day trip?`,
    expected: "outer_api",
    description: "Long paragraph about travel planning outside education system",
    role: "Student"
  },
  {
    input: `Please help me audit the platform before enrollment opens. 
    Verify that all services are running, confirm the database replication status, and check disk usage on the primary server. 
    After that, generate a quick analytics snapshot of last term's course completion rates so we can compare year over year.`,
    expected: "mixed",
    description: "Long paragraph combining system audit and analytics",
    role: "Admin"
  },
  {
    input: `I'm developing a review package for my calculus class and want it to feel cohesive. 
    Create a quiz that covers limits, derivatives, and integrals, then add a short set of application problems about optimization. 
    Also, give me a few conceptual questions that test intuition rather than computation.`,
    expected: "quiz_creation",
    description: "Long paragraph about a cohesive calculus quiz package",
    role: "Lecturer"
  }
];

async function runLongTextTests() {
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('TESTING LONG TEXT CLASSIFICATION');
  console.log('With Sentence-Level Analysis and Intent Voting');
  console.log('═══════════════════════════════════════════════════════════════\n');

  console.log('📋 Test Configuration:');
  console.log(`  • Long text threshold: ${FeatureEmbeddedClassifier.longTextThreshold} words`);
  console.log(`  • Minimum sentence length: ${FeatureEmbeddedClassifier.minSentenceLength} words`);
  console.log(`  • Routing threshold: ${(FeatureEmbeddedClassifier.routingThreshold * 100).toFixed(0)}%`);
  console.log('\n═══════════════════════════════════════════════════════════════\n');

  // Pre-load the classifier
  console.log('⏳ Loading Module Router...\n');
  await FeatureEmbeddedClassifier.getInstance();
  console.log('✓ Router ready!\n');

  for (let i = 0; i < longTextTests.length; i++) {
    const testCase = longTextTests[i];
    const wordCount = FeatureEmbeddedClassifier.countWords(testCase.input);
    
    console.log('═══════════════════════════════════════════════════════════════');
    console.log(`\n[Test ${i + 1}/${longTextTests.length}] ${testCase.description}`);
    console.log(`Word count: ${wordCount}`);
    console.log(`Role: ${testCase.role}`);
    console.log(`Expected category: ${testCase.expected}`);
    console.log('\nInput text:');
    console.log(`"${testCase.input}"`);
    console.log('\n─────────────────────────────────────────────────────────────');

    // Classify with role
    const result = await FeatureEmbeddedClassifier.classify(testCase.input, testCase.role);

    console.log('\n─────────────────────────────────────────────────────────────');
    console.log('ROUTING RESULT:');
    console.log(`  Decision: ${result.decision}`);
    console.log(`  Best Match: ${result.bestMatch.category} (${(result.bestMatch.score * 100).toFixed(2)}%)`);
    
    if (result.role) {
      console.log(`\n  Original Scores (before role boost):`);
      result.originalScores.forEach(({ category, score }) => {
        console.log(`      ${category.padEnd(25)}: ${(score * 100).toFixed(2)}%`);
      });
      console.log(`\n  Boosted Scores (with ${result.role} role):`);
      result.allScores.forEach(({ category, score }) => {
        const marker = score >= result.threshold ? '✓' : ' ';
        console.log(`    ${marker} ${category.padEnd(25)}: ${(score * 100).toFixed(2)}%`);
      });
    } else {
      console.log(`\n  All Scores:`);
      result.allScores.forEach(({ category, score }) => {
        const marker = score >= result.threshold ? '✓' : ' ';
        console.log(`    ${marker} ${category.padEnd(25)}: ${(score * 100).toFixed(2)}%`);
      });
    }

    // Evaluation
    let evaluation = '';
    if (testCase.expected === 'mixed') {
      evaluation = '⚠ MIXED INTENT (multiple categories expected)';
    } else if (testCase.expected === 'gray_zone') {
      evaluation = '⚠ GRAY ZONE (boundary case)';
    } else if (testCase.expected === 'outer_api') {
      if (result.decision === 'forward to outer API') {
        evaluation = '✓ CORRECT (Routed to outer API)';
      } else {
        evaluation = '✗ INCORRECT (Should route to outer API)';
      }
    } else {
      if (result.decision.includes(testCase.expected)) {
        evaluation = `✓ CORRECT (Routed to ${testCase.expected} module)`;
      } else {
        evaluation = `✗ INCORRECT (Expected: ${testCase.expected}, Got: ${result.bestMatch.category})`;
      }
    }
    
    console.log(`\n${evaluation}`);
    console.log('\n═══════════════════════════════════════════════════════════════\n');
    
    // Add delay to avoid overwhelming the console
    await new Promise(resolve => setTimeout(resolve, 500));
  }

  console.log('═══════════════════════════════════════════════════════════════');
  console.log('TESTING COMPLETE');
  console.log('═══════════════════════════════════════════════════════════════\n');
}

// Run the tests
runLongTextTests().catch(console.error);
