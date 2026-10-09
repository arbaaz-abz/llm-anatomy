// The paged-attention concept module: re-exports the lesson and mounts it with the track's data (template rule 1).
import { mountLesson } from '@shared/lesson-page.js';
import { LESSON, lessonFor } from './paged-attention/content.js';

export { LESSON };
export default { slug: 'paged-attention', title: 'PagedAttention', mount: (el, ctx) => mountLesson(el, ctx, lessonFor(ctx?.data)) };
