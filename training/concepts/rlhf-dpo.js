// rlhf-dpo concept module: re-exports the lesson and mounts it with the track's data (dated text needs ctx.data).
import { mountLesson } from '@shared/lesson-page.js';
import { LESSON, lessonFor } from './rlhf-dpo/content.js';

export { LESSON };
export default { slug: 'rlhf-dpo', title: 'RLHF and DPO', mount: (el, ctx) => mountLesson(el, ctx, lessonFor(ctx?.data)) };
