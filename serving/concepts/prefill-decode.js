// prefill-decode concept module: re-exports the lesson and mounts it with the track's data (dated text and the toy need ctx.data).
import { mountLesson } from '@shared/lesson-page.js';
import { LESSON, lessonFor } from './prefill-decode/content.js';

export { LESSON };
export default { slug: 'prefill-decode', title: 'Prefill vs decode', mount: (el, ctx) => mountLesson(el, ctx, lessonFor(ctx?.data)) };
