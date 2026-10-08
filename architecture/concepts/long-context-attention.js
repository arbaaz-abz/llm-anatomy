// The long-context-attention concept module: re-exports the lesson and mounts it with the track's data (template rule 1).
import { mountLesson } from '@shared/lesson-page.js';
import { LESSON, lessonFor } from './long-context-attention/content.js';

export { LESSON };
export default { slug: 'long-context-attention', title: 'Reaching 1M tokens', mount: (el, ctx) => mountLesson(el, ctx, lessonFor(ctx?.data)) };
