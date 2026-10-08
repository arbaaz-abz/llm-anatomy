// gpu-primer concept module: re-exports the lesson and mounts it with the track's data (dated text and the toy need ctx.data).
import { mountLesson } from '@shared/lesson-page.js';
import { LESSON, lessonFor } from './gpu-primer/content.js';

export { LESSON };
export default { slug: 'gpu-primer', title: 'A GPU for LLM people', mount: (el, ctx) => mountLesson(el, ctx, lessonFor(ctx?.data)) };
