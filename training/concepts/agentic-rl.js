// The agentic-rl concept module: re-exports the lesson and mounts it with the track's data (template rule 1).
import { mountLesson } from '@shared/lesson-page.js';
import { LESSON, lessonFor } from './agentic-rl/content.js';

export { LESSON };
export default { slug: 'agentic-rl', title: 'Agentic RL', mount: (el, ctx) => mountLesson(el, ctx, lessonFor(ctx?.data)) };
