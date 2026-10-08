// training-pipeline concept module: re-exports the lesson and mounts it with the track's data (dated text needs ctx.data).
import { mountLesson } from '@shared/lesson-page.js';
import { LESSON, lessonFor } from './training-pipeline/content.js';

export { LESSON };
export default { slug: 'training-pipeline', title: 'The 2026 training pipeline', mount: (el, ctx) => mountLesson(el, ctx, lessonFor(ctx?.data)) };
