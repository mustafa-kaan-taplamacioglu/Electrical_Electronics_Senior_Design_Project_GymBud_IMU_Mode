import OpenAI from 'openai';
import type { RepData, ExerciseType } from '../types';
import { EXERCISES } from '../config/exercises';

let openaiClient: OpenAI | null = null;

export const initOpenAI = (apiKey: string) => {
  openaiClient = new OpenAI({
    apiKey,
    dangerouslyAllowBrowser: true, // For client-side usage
  });
};

export const getRepFeedback = async (
  exercise: ExerciseType,
  repData: RepData
): Promise<string> => {
  if (!openaiClient) {
    return 'API key not configured';
  }

  const exerciseConfig = EXERCISES[exercise];
  const issuesText = repData.issues.length > 0 
    ? repData.issues.join(', ') 
    : 'None';

  const prompt = `You are a fitness coach. The user just completed a ${exerciseConfig.displayName} rep.

Rep #${repData.repNumber}
Form Score: ${repData.formScore.toFixed(0)}%
Detected Issues: ${issuesText}

Give SHORT and MOTIVATING feedback (max 2 sentences).
- Start positive
- Suggest 1 correction if needed
- Be friendly and encouraging`;

  try {
    const response = await openaiClient.chat.completions.create({
      model: 'gpt-3.5-turbo',
      messages: [{ role: 'user', content: prompt }],
      max_tokens: 300,
      temperature: 0.7,
    });

    return response.choices[0]?.message?.content || 'Could not get feedback';
  } catch (error) {
    console.error('OpenAI error:', error);
    return 'Error getting feedback';
  }
};

export const getSessionFeedback = async (
  exercise: ExerciseType,
  reps: RepData[],
  durationSeconds: number
): Promise<string> => {
  if (!openaiClient) {
    return 'API key not configured';
  }

  const exerciseConfig = EXERCISES[exercise];
  const avgScore = reps.reduce((sum, r) => sum + r.formScore, 0) / reps.length;
  const bestScore = Math.max(...reps.map(r => r.formScore));
  const worstScore = Math.min(...reps.map(r => r.formScore));
  
  // Count issues
  const issueCount: Record<string, number> = {};
  reps.forEach(r => {
    r.issues.forEach(issue => {
      issueCount[issue] = (issueCount[issue] || 0) + 1;
    });
  });
  const topIssues = Object.entries(issueCount)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([issue]) => issue);

  const prompt = `You are a fitness coach. The user completed a ${exerciseConfig.displayName} workout.

📊 WORKOUT SUMMARY:
- Total Reps: ${reps.length}
- Duration: ${Math.floor(durationSeconds / 60)} min ${Math.floor(durationSeconds % 60)} sec
- Average Form: ${avgScore.toFixed(0)}%
- Best Form: ${bestScore.toFixed(0)}%
- Worst Form: ${worstScore.toFixed(0)}%
- Common Issues: ${topIssues.length > 0 ? topIssues.join(', ') : 'None'}

Give detailed feedback:
1. Overall performance assessment
2. Improvement suggestions (if any)
3. Motivating closing

Be friendly and encouraging. Max 4-5 sentences.`;

  try {
    const response = await openaiClient.chat.completions.create({
      model: 'gpt-4',
      messages: [{ role: 'user', content: prompt }],
      max_tokens: 500,
      temperature: 0.7,
    });

    return response.choices[0]?.message?.content || 'Could not get feedback';
  } catch (error) {
    console.error('OpenAI error:', error);
    return 'Error getting feedback';
  }
};
