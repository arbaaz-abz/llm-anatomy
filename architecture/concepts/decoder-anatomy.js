// decoder-anatomy concept module: re-exports the lesson and mounts it with the track's data (dated text needs ctx.data).
import { mountLesson } from '@shared/lesson-page.js';
import { LESSON, lessonFor } from './decoder-anatomy/content.js';

export { LESSON };
export default { slug: 'decoder-anatomy', title: 'The whole model, end to end', mount: (el, ctx) => mountLesson(el, ctx, lessonFor(ctx?.data)) };
