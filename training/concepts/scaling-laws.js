// The scaling-laws concept module: re-exports the lesson and mounts it with the track's data (template rule 1).
import { mountLesson } from '@shared/lesson-page.js';
import { LESSON, lessonFor } from './scaling-laws/content.js';

export { LESSON };
export default { slug: 'scaling-laws', title: 'Scaling laws', mount: (el, ctx) => mountLesson(el, ctx, lessonFor(ctx?.data)) };
