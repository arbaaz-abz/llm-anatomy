// serving-calculator concept module: re-exports the lesson and mounts it with the track's data (dated text and the toy need ctx.data).
import { mountLesson } from '@shared/lesson-page.js';
import { LESSON, lessonFor } from './serving-calculator/content.js';

export { LESSON };
export default { slug: 'serving-calculator', title: 'Serving a 1T model', mount: (el, ctx) => mountLesson(el, ctx, lessonFor(ctx?.data)) };
