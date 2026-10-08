// rlvr-grpo concept module: re-exports the lesson and mounts it with the track's data (dated text needs ctx.data).
import { mountLesson } from '@shared/lesson-page.js';
import { LESSON, lessonFor } from './rlvr-grpo/content.js';

export { LESSON };
export default { slug: 'rlvr-grpo', title: 'RLVR and GRPO', mount: (el, ctx) => mountLesson(el, ctx, lessonFor(ctx?.data)) };
