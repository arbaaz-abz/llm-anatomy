// The decoder-recap concept module: re-exports the lesson and mounts it with the track's data (template rule 1).
import { mountLesson } from '@shared/lesson-page.js';
import { LESSON, lessonFor } from './decoder-recap/content.js';

export { LESSON };
export default { slug: 'decoder-recap', title: 'From GPT-3 to 2026', mount: (el, ctx) => mountLesson(el, ctx, lessonFor(ctx?.data)) };
