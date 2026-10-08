// The sampling concept module: re-exports the lesson and mounts it with the track's data (template rule 1).
import { mountLesson } from '@shared/lesson-page.js';
import { LESSON, lessonFor } from './sampling/content.js';

export { LESSON };
export default { slug: 'sampling', title: 'Picking the next token', mount: (el, ctx) => mountLesson(el, ctx, lessonFor(ctx?.data)) };
