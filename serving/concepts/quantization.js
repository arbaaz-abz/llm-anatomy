// quantization concept module: re-exports the lesson and mounts it with the track's data (dated text and the toy need ctx.data).
import { mountLesson } from '@shared/lesson-page.js';
import { LESSON, lessonFor } from './quantization/content.js';

export { LESSON };
export default { slug: 'quantization', title: 'Quantization', mount: (el, ctx) => mountLesson(el, ctx, lessonFor(ctx?.data)) };
