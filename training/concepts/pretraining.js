// pretraining concept module: re-exports the lesson and mounts it with the track's data (dated text needs ctx.data).
import { mountLesson } from '@shared/lesson-page.js';
import { LESSON, lessonFor } from './pretraining/content.js';

export { LESSON };
export default { slug: 'pretraining', title: 'Pretraining', mount: (el, ctx) => mountLesson(el, ctx, lessonFor(ctx?.data)) };
