
import http from 'http';
import querystring from 'querystring';
import url from 'url';

import { NLIClassificationPipeline, labels } from './NLI.js';
// import { FeatureEmbeddedClassifier } from './FeatureEmbedded.js';
// import { FeatureEmbeddedClassifier } from './FeatureEmbeddedCentroid.js';
// import { FeatureEmbeddedClassifier } from './ModuleRouter.js';
import { FeatureEmbeddedClassifier } from './ModuleRouterLong.js';


// Choose which classifier to use: 'nli' or 'few-shot'
// const CLASSIFIER_TYPE = 'nli';
const CLASSIFIER_TYPE = 'few-shot';
// Pre-load the model
if (CLASSIFIER_TYPE === 'nli') {
  NLIClassificationPipeline.getInstance();
} else {
  FeatureEmbeddedClassifier.getInstance();
}

// Define the HTTP server
const server = http.createServer();
const hostname = '127.0.0.1';
const port = 3000;

// Listen for requests made to the server
server.on('request', async (req, res) => {
  // Parse the request URL
  const parsedUrl = url.parse(req.url);

  // Set the response headers
  res.setHeader('Content-Type', 'application/json');

  let response;
  
  if (parsedUrl.pathname === '/classify' && req.method === 'POST') {
    // Parse JSON body
    let body = '';
    req.on('data', chunk => {
      body += chunk.toString();
    });
    
    req.on('end', async () => {
      try {
        const { text, role } = JSON.parse(body);
        
        if (text) {
          // Use the selected classifier
          if (CLASSIFIER_TYPE === 'nli') {
            const classifier = await NLIClassificationPipeline.getInstance();
            response = await classifier(text, labels);
          } else {
            response = await FeatureEmbeddedClassifier.classify(text, role);
          }
          
          // Console log the results with percentages
          console.log('\n=== Classification Results ===');
          console.log(`Classifier: ${CLASSIFIER_TYPE.toUpperCase()}`);
          console.log(`Text: "${text}"`);
          console.log('\nLabel Scores:');
          
          // Handle different response formats
          if (response.decision) {
            // ModuleRouter format
            console.log(`\nDecision: ${response.decision}`);
            console.log(`Threshold: ${(response.threshold * 100).toFixed(0)}%\n`);
            console.log('All Scores:');
            response.allScores.forEach(({ category, score }) => {
              const percentage = (score * 100).toFixed(2);
              const marker = score >= response.threshold ? '✓' : ' ';
              console.log(`  ${marker} ${category.padEnd(45)} : ${percentage}%`);
            });
          } else if (response.labels && response.scores) {
            // Standard format (NLI, FeatureEmbedded)
            response.labels.forEach((label, index) => {
              const percentage = (response.scores[index] * 100).toFixed(2);
              console.log(`  ${label.padEnd(50)} : ${percentage}%`);
            });
          }
          console.log('=============================\n');
          
          res.statusCode = 200;
        } else {
          response = { 'error': 'Missing text field in request body' };
          res.statusCode = 400;
        }
      } catch (error) {
        response = { 'error': 'Invalid JSON body' };
        res.statusCode = 400;
      }
      
      // Send the JSON response
      res.end(JSON.stringify(response));
    });
  } else {
    response = { 'error': 'Bad request. Use POST /classify with JSON body' };
    res.statusCode = 400;
    res.end(JSON.stringify(response));
  }
});

server.listen(port, hostname, () => {
  console.log(`Server running at http://${hostname}:${port}/`);
});
