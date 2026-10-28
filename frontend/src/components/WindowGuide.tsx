import { Descriptions, Divider } from 'antd';
import guideSource from '../content/window-guide.md?raw';
import MarkdownDoc from './MarkdownDoc';

type Props = {
  windowS?: number;
  strideS?: number;
  threshold?: number;
  fps?: number;
};

/**
 * "计算原理" tab: the current job's scoring parameters on top, followed by the
 * long-form documentation in `src/content/window-guide.md`.
 *
 * The document is plain markdown (rendered with GFM + KaTeX), so it can be
 * edited without touching JSX.
 */
export default function WindowGuide({ windowS, strideS, threshold, fps }: Props) {
  const w = windowS && windowS > 0 ? windowS : 0.5;
  const st = strideS && strideS > 0 ? strideS : w * 0.5;
  const th = threshold && threshold > 0 ? threshold : 0.5;
  const f = fps && fps > 0 ? fps : 25;
  // Mirrors features::windows_for_track: frames per window / per stride.
  const n = Math.max(2, Math.round(w * f));
  const m = Math.max(1, Math.round(st * f));

  return (
    <div className="window-guide">
      <Descriptions
        size="small"
        bordered
        column={{ xs: 1, sm: 2, lg: 4 }}
        items={[
          {
            key: 'fps',
            label: '帧率',
            children: `${Number.isInteger(f) ? f : f.toFixed(2)} fps`,
          },
          { key: 'window', label: '窗口长度', children: `${w.toFixed(2)} s ≈ ${n} 帧` },
          { key: 'stride', label: '步长', children: `${st.toFixed(2)} s ≈ ${m} 帧` },
          {
            key: 'threshold',
            label: '判定阈值',
            children: `p(immobile) > ${th.toFixed(2)}`,
          },
        ]}
      />
      <Divider style={{ margin: '16px 0' }} />
      <MarkdownDoc source={guideSource} />
    </div>
  );
}
